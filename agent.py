"""
LiveKit Agents 1.x worker for the Ezee Flights outbound voice agent (Zaya).

Loads all prompts, models, voice settings, STT keyterms, turn-taking parameters,
and call limits at runtime from `spec/flight_booking.json` + `prompt_overrides.yaml`.

Usage:
  # Console mode (talk via laptop mic & speakers, no SIP required):
  python agent.py console

  # Dev worker mode (connects to LiveKit Cloud / self-hosted server):
  python agent.py dev

  # Production worker mode:
  python agent.py start
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import random
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import Field

from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import elevenlabs, openai, silero

from post_call import generate_and_save_post_call_summary
from providers.flights import FlightProvider, create_flight_provider
from providers.stt import build_stt
from spec_loader import AgentSpec, CallMetadata, load_spec

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger("ezee-flights.agent")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Load spec + overrides at runtime
SPEC: AgentSpec = load_spec()


def _build_openai_llm(spec: AgentSpec) -> openai.LLM:
    """
    Build the OpenAI LLM from the `OpenAI Brain` node:
    model = gpt-5.4-mini, max_completion_tokens = 500, reasoning_effort = none.
    """
    sig = inspect.signature(openai.LLM.__init__).parameters
    kwargs: Dict[str, Any] = {
        "model": spec.llm.model,
    }
    if "max_completion_tokens" in sig:
        kwargs["max_completion_tokens"] = spec.llm.max_completion_tokens
    if "reasoning_effort" in sig and spec.llm.reasoning_effort:
        kwargs["reasoning_effort"] = spec.llm.reasoning_effort
    return openai.LLM(**kwargs)


def _build_elevenlabs_tts(spec: AgentSpec) -> elevenlabs.TTS:
    """
    Build ElevenLabs TTS from the `ElevenLabs Voice` node:
    model = eleven_flash_v2_5, voice_id = ELEVEN_VOICE_ID or ack0QsRaQyDLnVyMQTSd,
    stability = 0.4, similarity_boost = 0.8, style = 0.35, use_speaker_boost = True, speed = 1.0.
    """
    tts_sig = inspect.signature(elevenlabs.TTS.__init__).parameters
    kwargs: Dict[str, Any] = {
        "model": spec.tts.model,
    }
    eleven_key = (os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY") or "").strip()
    if eleven_key and "api_key" in tts_sig:
        kwargs["api_key"] = eleven_key
    if "voice_id" in tts_sig:
        kwargs["voice_id"] = spec.tts.voice_id
    elif "voice" in tts_sig:
        kwargs["voice"] = spec.tts.voice_id

    if hasattr(elevenlabs, "VoiceSettings"):
        vs_sig = inspect.signature(elevenlabs.VoiceSettings.__init__).parameters
        vs_kwargs: Dict[str, Any] = {
            "stability": spec.tts.stability,
            "similarity_boost": spec.tts.similarity_boost,
        }
        if "style" in vs_sig:
            vs_kwargs["style"] = spec.tts.style
        if "use_speaker_boost" in vs_sig:
            vs_kwargs["use_speaker_boost"] = spec.tts.use_speaker_boost
        if "speed" in vs_sig:
            vs_kwargs["speed"] = spec.tts.speed
        kwargs["voice_settings"] = elevenlabs.VoiceSettings(**vs_kwargs)

    return elevenlabs.TTS(**kwargs)


def _build_turn_detector() -> Any:
    """Load the LiveKit multilingual turn detector if installed."""
    try:
        from livekit.plugins.turn_detector.multilingual import MultilingualModel

        return MultilingualModel()
    except Exception as exc:
        logger.warning("Multilingual turn detector unavailable (%s); falling back to VAD turn detection.", exc)
        return None


def build_ezee_agent(
    spec: AgentSpec,
    metadata: CallMetadata,
    flight_provider: FlightProvider,
) -> Agent:
    """
    Construct the LiveKit `Agent` with:
    - System prompt = `Start.masterInstruction` + `eandAI.instructions` (placeholders filled)
    - `@function_tool` `search_flights(query: str)` using `SearchFlights.instructions` verbatim
    """
    system_prompt = spec.build_system_prompt(metadata)
    tool_description = spec.search_tool.instructions
    query_description = spec.search_tool.query_description

    @function_tool(
        name="search_flights",
        description=tool_description,
    )
    async def search_flights(
        context: RunContext,
        query: Annotated[str, Field(description=query_description)],
    ) -> str:
        logger.info("Tool call search_flights(query=%r)", query)
        # Optional subtle filler before tool execution if configured via ENV
        if (
            spec.fillers.enabled
            and spec.fillers.fillers
            and os.getenv("TOOL_FILLER_AUDIO", "false").lower() == "true"
        ):
            filler_word = random.choice(spec.fillers.fillers)
            try:
                context.session.say(f"{filler_word.capitalize()}, one moment.", add_to_chat_ctx=False)
            except Exception:
                pass

        result = await flight_provider.search(query)
        logger.info("search_flights returned %d chars", len(result))
        return result

    return Agent(
        instructions=system_prompt,
        tools=[search_flights],
    )


def prewarm(proc: JobProcess) -> None:
    """Preload Silero VAD into process userdata for fast call startup."""
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["spec"] = load_spec()
    proc.userdata["flight_provider"] = create_flight_provider(spec=proc.userdata["spec"])


async def _hangup_call(ctx: JobContext, session: AgentSession, reason: str) -> None:
    """Gracefully hang up the call / delete the LiveKit room."""
    logger.info("Hanging up call (reason=%s)", reason)
    try:
        await session.aclose()
    except Exception:
        pass
    try:
        if ctx.room and ctx.room.name and ctx.api:
            await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
    except Exception:
        pass


async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint for each inbound/outbound LiveKit call or console session."""
    spec: AgentSpec = ctx.proc.userdata.get("spec") or load_spec()
    flight_provider: FlightProvider = (
        ctx.proc.userdata.get("flight_provider") or create_flight_provider(spec=spec)
    )
    vad: silero.VAD = ctx.proc.userdata.get("vad") or silero.VAD.load()

    # Parse job metadata (or fall back to room metadata / environment defaults for console mode)
    raw_meta = (ctx.job.metadata if ctx.job else "") or ""
    metadata = CallMetadata.from_json(raw_meta)
    if not metadata.resolved_name() and os.getenv("DEFAULT_CALLER_NAME"):
        metadata.caller_name = os.getenv("DEFAULT_CALLER_NAME", "")
        metadata.customer_name = metadata.caller_name
    if not metadata.enquiry_route and os.getenv("DEFAULT_ENQUIRY_ROUTE"):
        metadata.enquiry_route = os.getenv("DEFAULT_ENQUIRY_ROUTE", "")
    if not metadata.phone_number and os.getenv("DEFAULT_PHONE_NUMBER"):
        metadata.phone_number = os.getenv("DEFAULT_PHONE_NUMBER", "")

    call_id = (
        (ctx.room.name if ctx.room and ctx.room.name else None)
        or (ctx.job.id if ctx.job and ctx.job.id else None)
        or f"call-{uuid.uuid4().hex[:8]}"
    )
    start_time = time.monotonic()
    transcript_log: List[Dict[str, Any]] = []
    termination_reason = "completed"
    summary_written = False

    async def _write_post_call_once() -> None:
        nonlocal summary_written
        if summary_written:
            return
        summary_written = True
        duration = time.monotonic() - start_time
        await generate_and_save_post_call_summary(
            call_id=call_id,
            transcript=transcript_log,
            metadata=metadata,
            spec=spec,
            duration_seconds=duration,
            termination_reason=termination_reason,
        )

    ctx.add_shutdown_callback(_write_post_call_once)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Check room metadata if job metadata was empty
    if not raw_meta and ctx.room and ctx.room.metadata:
        room_meta = CallMetadata.from_json(ctx.room.metadata)
        if room_meta.resolved_name():
            metadata.caller_name = room_meta.caller_name
            metadata.customer_name = room_meta.customer_name
        if room_meta.enquiry_route:
            metadata.enquiry_route = room_meta.enquiry_route
        if room_meta.phone_number:
            metadata.phone_number = room_meta.phone_number

    logger.info(
        "Starting session call_id=%s caller=%r route=%r phone=%r",
        call_id,
        metadata.resolved_name(),
        metadata.enquiry_route,
        metadata.phone_number,
    )

    # Dial outbound SIP participant if phone_number is specified and not yet dialed
    sip_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID", "").strip()
    if metadata.phone_number and sip_trunk_id and ctx.room and ctx.api:
        existing_sip = [
            p for p in ctx.room.remote_participants.values() if p.identity.startswith("sip_")
        ]
        if not existing_sip:
            logger.info(
                "Dialing outbound SIP participant %s via trunk %s",
                metadata.phone_number,
                sip_trunk_id,
            )
            try:
                await ctx.api.sip.create_sip_participant(
                    api.CreateSIPParticipantRequest(
                        room_name=ctx.room.name,
                        sip_trunk_id=sip_trunk_id,
                        sip_call_to=metadata.phone_number,
                        participant_identity=f"sip_{metadata.phone_number}",
                        participant_name=metadata.resolved_name() or "Customer",
                        wait_until_answered=True,
                    )
                )
            except Exception as exc:
                logger.error("Failed to dial outbound SIP participant: %s", exc)
                termination_reason = f"sip_dial_failed: {exc}"
                await _write_post_call_once()
                return

    # Build STT, LLM, TTS, Turn Detector, and AgentSession
    stt = build_stt(spec.stt)
    llm = _build_openai_llm(spec)
    tts = _build_elevenlabs_tts(spec)
    turn_detector = _build_turn_detector()

    session_sig = inspect.signature(AgentSession.__init__).parameters
    session_kwargs: Dict[str, Any] = {
        "stt": stt,
        "llm": llm,
        "tts": tts,
        "vad": vad,
        "min_interruption_words": spec.turn_taking.min_interruption_words,
        "max_endpointing_delay": spec.turn_taking.max_endpointing_delay,
    }
    if turn_detector is not None and "turn_detection" in session_sig:
        session_kwargs["turn_detection"] = turn_detector

    session = AgentSession(**session_kwargs)
    agent = build_ezee_agent(spec=spec, metadata=metadata, flight_provider=flight_provider)

    # Activity tracking for the 30s silence timeout
    last_activity_ts = time.monotonic()
    agent_speaking = False
    user_speaking = False

    def _touch_activity() -> None:
        nonlocal last_activity_ts
        last_activity_ts = time.monotonic()

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: Any) -> None:
        nonlocal user_speaking
        _touch_activity()
        new_state = getattr(ev, "new_state", "")
        user_speaking = new_state == "speaking"

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: Any) -> None:
        nonlocal agent_speaking
        _touch_activity()
        new_state = getattr(ev, "new_state", "")
        agent_speaking = new_state == "speaking"

    @session.on("conversation_item_added")
    def _on_conversation_item(ev: Any) -> None:
        _touch_activity()
        item = getattr(ev, "item", None)
        if item is None:
            return
        role = str(getattr(item, "role", "unknown"))
        text_content = getattr(item, "text_content", None)
        if callable(text_content):
            text_content = text_content()
        if not text_content:
            content_list = getattr(item, "content", [])
            if isinstance(content_list, list):
                text_content = " ".join(str(c) for c in content_list if isinstance(c, str))
        if text_content:
            transcript_log.append(
                {
                    "role": role,
                    "content": str(text_content),
                    "timestamp": round(time.monotonic() - start_time, 2),
                }
            )

    # Start the agent session
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )

    # Start office ambience background audio at 20% volume (`BGNoise_Office`)
    bg_player: Optional[BackgroundAudioPlayer] = None
    if spec.bg_noise.enabled:
        try:
            bg_player = BackgroundAudioPlayer(
                ambient_sound=AudioClipConfig(spec.bg_noise.volume_float)
            )
            await bg_player.start(room=ctx.room, agent_session=session)
            logger.info(
                "Started BGNoise_Office (%s) at %d%% volume",
                spec.bg_noise.track,
                spec.bg_noise.volume_percent,
            )
        except Exception as exc:
            logger.warning("Could not start BackgroundAudioPlayer (%s)", exc)

    # Watchdog task for maxCallDurationMinutes (10m) and silenceTimeoutSeconds (30s)
    max_duration_sec = (
        spec.limits.max_call_duration_minutes * 60.0
        if spec.limits.max_call_duration_enabled
        else float("inf")
    )
    silence_timeout_sec = spec.limits.silence_timeout_seconds

    async def _call_limits_watchdog() -> None:
        nonlocal termination_reason
        try:
            while True:
                await asyncio.sleep(1.0)
                now = time.monotonic()
                elapsed = now - start_time

                if elapsed >= max_duration_sec:
                    termination_reason = "max_call_duration_reached"
                    logger.info(
                        "Max call duration (%.1f min) reached; hanging up gracefully.",
                        spec.limits.max_call_duration_minutes,
                    )
                    try:
                        await session.say(
                            "I want to be respectful of your time, so I will have one of our travel consultants follow up with you directly. Thank you for speaking with Ezee Flights, goodbye.",
                            allow_interruptions=False,
                        )
                    except Exception:
                        pass
                    await _hangup_call(ctx, session, termination_reason)
                    break

                if not agent_speaking and not user_speaking:
                    idle_sec = now - last_activity_ts
                    if idle_sec >= silence_timeout_sec:
                        termination_reason = "silence_timeout"
                        logger.info(
                            "Silence timeout (%.1fs) reached; hanging up gracefully.",
                            silence_timeout_sec,
                        )
                        try:
                            await session.say(
                                "It seems we may have lost connection, so I will let you go for now and we can try you again later. Thank you, goodbye.",
                                allow_interruptions=False,
                            )
                        except Exception:
                            pass
                        await _hangup_call(ctx, session, termination_reason)
                        break
        except asyncio.CancelledError:
            pass

    watchdog_task = asyncio.create_task(_call_limits_watchdog())

    # Ensure post-call summary is written if session closes in console mode
    @session.on("close")
    def _on_session_close(_ev: Any) -> None:
        watchdog_task.cancel()
        if bg_player is not None:
            asyncio.create_task(bg_player.aclose())
        asyncio.create_task(_write_post_call_once())


def AudioClipConfig(volume: float) -> Any:
    """Create the AudioConfig for BuiltinAudioClip.OFFICE_AMBIENCE at the given volume."""
    try:
        from livekit.agents import AudioConfig

        return AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=volume)
    except ImportError:
        return BuiltinAudioClip.OFFICE_AMBIENCE


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=os.getenv("LIVEKIT_AGENT_NAME", "ezee-flights-outbound"),
        )
    )
