"""
Post-call transcript analysis and JSON summary writer.

Uses `Start.parameters.post_call_model` (gpt-5.4-mini) from `spec/flight_booking.json`
to summarize every call into `calls/<call_id>.json` with:
- outcome
- chosen_option
- callback_time
- notes_for_consultant
- metadata & full transcript
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from spec_loader import AgentSpec, CallMetadata, load_spec

logger = logging.getLogger("ezee-flights.post-call")

CALLS_DIR = Path(__file__).parent / "calls"

POST_CALL_SYSTEM_PROMPT = """You are the post-call analysis model for Ezee Flights.
Analyze the outbound call transcript between Zaya (Ezee Flights assistant) and the customer.
Return strictly a JSON object with the following keys:
- "outcome": One of ["ready_to_book", "callback_requested", "send_options", "existing_booking_or_refund", "opted_out_do_not_call", "wrong_number", "not_interested", "incomplete"]
- "chosen_option": The specific flight option(s) the caller selected or showed strongest interest in (airline, route, dates, quoted fare), or null if none was chosen.
- "callback_time": Agreed callback / follow-up day and time if mentioned, or null.
- "notes_for_consultant": Clear, actionable handover notes for the human Ezee Flights travel consultant (passengers, ages, route, dates, date flexibility, baggage needs, budget or price objections, visa questions, booking reference if existing booking, or opt-out confirmation). Never include raw card or passport numbers.
"""


async def generate_and_save_post_call_summary(
    call_id: str,
    transcript: List[Dict[str, Any]],
    metadata: Optional[CallMetadata] = None,
    spec: Optional[AgentSpec] = None,
    duration_seconds: Optional[float] = None,
    termination_reason: str = "completed",
    calls_dir: Optional[Path] = None,
) -> Path:
    """
    Send the call transcript to OpenAI (`post_call_model`) and write the resulting
    summary JSON to `calls/<call_id>.json`.
    """
    loaded_spec = spec or load_spec()
    meta = metadata or CallMetadata()
    out_dir = calls_dir or CALLS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    formatted_lines: List[str] = []
    for item in transcript:
        role = item.get("role", "unknown").upper()
        content = item.get("content", "")
        if content:
            formatted_lines.append(f"{role}: {content}")
    transcript_text = "\n".join(formatted_lines) if formatted_lines else "(No speech recorded)"

    summary_data: Dict[str, Any] = {
        "outcome": "incomplete",
        "chosen_option": None,
        "callback_time": None,
        "notes_for_consultant": "Call ended before a full summary could be generated.",
    }

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key and formatted_lines:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        model_name = os.getenv("POST_CALL_MODEL") or loaded_spec.post_call.model
        user_prompt = (
            f"Call Metadata:\n"
            f"- Caller Name: {meta.resolved_name() or 'Unknown'}\n"
            f"- Enquiry Route: {meta.enquiry_route or 'Unspecified'}\n"
            f"- Phone Number: {meta.phone_number or 'Unspecified'}\n"
            f"- Termination Reason: {termination_reason}\n\n"
            f"Transcript:\n{transcript_text}"
        )
        try:
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": POST_CALL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            }
            # Include reasoning_effort if supported by the model
            if loaded_spec.post_call.reasoning_effort and loaded_spec.post_call.reasoning_effort != "none":
                kwargs["reasoning_effort"] = loaded_spec.post_call.reasoning_effort

            try:
                resp = await client.chat.completions.create(**kwargs)
            except Exception:
                # Fallback without reasoning_effort if model rejects the parameter
                kwargs.pop("reasoning_effort", None)
                resp = await client.chat.completions.create(**kwargs)

            raw_content = resp.choices[0].message.content or "{}"
            parsed = json.loads(raw_content)
            if isinstance(parsed, dict):
                summary_data.update(
                    {
                        "outcome": parsed.get("outcome", "incomplete"),
                        "chosen_option": parsed.get("chosen_option"),
                        "callback_time": parsed.get("callback_time"),
                        "notes_for_consultant": parsed.get("notes_for_consultant", ""),
                    }
                )
        except Exception as exc:
            logger.warning("OpenAI post-call summary failed (%s); using heuristic fallback.", exc)
            summary_data = _heuristic_summary(transcript, meta, termination_reason)
    else:
        summary_data = _heuristic_summary(transcript, meta, termination_reason)

    payload: Dict[str, Any] = {
        "call_id": call_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration_seconds, 2) if duration_seconds is not None else None,
        "termination_reason": termination_reason,
        "post_call_model": loaded_spec.post_call.model,
        "metadata": {
            "caller_name": meta.caller_name,
            "customer_name": meta.resolved_name(),
            "enquiry_route": meta.enquiry_route,
            "phone_number": meta.phone_number,
        },
        "outcome": summary_data.get("outcome"),
        "chosen_option": summary_data.get("chosen_option"),
        "callback_time": summary_data.get("callback_time"),
        "notes_for_consultant": summary_data.get("notes_for_consultant"),
        "transcript": transcript,
    }

    safe_call_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in call_id)
    out_path = out_dir / f"{safe_call_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved post-call summary to %s", out_path)
    return out_path


def save_post_call_summary_sync(
    call_id: str,
    transcript: List[Dict[str, Any]],
    metadata: Optional[CallMetadata] = None,
    spec: Optional[AgentSpec] = None,
    duration_seconds: Optional[float] = None,
    termination_reason: str = "completed",
    calls_dir: Optional[Path] = None,
) -> Path:
    """Synchronous post-call summary writer (used for offline test runs)."""
    loaded_spec = spec or load_spec()
    meta = metadata or CallMetadata()
    out_dir = calls_dir or CALLS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_data = _heuristic_summary(transcript, meta, termination_reason)
    payload: Dict[str, Any] = {
        "call_id": call_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration_seconds, 2) if duration_seconds is not None else None,
        "termination_reason": termination_reason,
        "post_call_model": loaded_spec.post_call.model,
        "metadata": {
            "caller_name": meta.caller_name,
            "customer_name": meta.resolved_name(),
            "enquiry_route": meta.enquiry_route,
            "phone_number": meta.phone_number,
        },
        "outcome": summary_data.get("outcome"),
        "chosen_option": summary_data.get("chosen_option"),
        "callback_time": summary_data.get("callback_time"),
        "notes_for_consultant": summary_data.get("notes_for_consultant"),
        "transcript": transcript,
    }
    safe_call_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in call_id)
    out_path = out_dir / f"{safe_call_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def _heuristic_summary(
    transcript: List[Dict[str, Any]],
    meta: CallMetadata,
    termination_reason: str,
) -> Dict[str, Any]:
    """Fallback rule-based summary if OpenAI API is unreachable."""
    full_user_text = " ".join(
        t.get("content", "").lower() for t in transcript if t.get("role") == "user"
    )
    full_all_text = " ".join(t.get("content", "") for t in transcript)

    outcome = "incomplete"
    chosen_option: Optional[str] = None
    callback_time: Optional[str] = None

    if any(p in full_user_text for p in ("don't call", "do not call", "remove my number", "take me off", "stop calling")):
        outcome = "opted_out_do_not_call"
    elif "refund" in full_user_text or "existing booking" in full_user_text:
        outcome = "existing_booking_or_refund"
    elif any(p in full_user_text for p in ("connect me", "book the", "let's book", "go with emirates", "go with turkish", "go with gulf", "take the direct", "take the turkish")):
        outcome = "ready_to_book"
    elif any(p in full_user_text for p in ("whatsapp", "send me", "call me back", "tomorrow", "later today", "evening")):
        outcome = "callback_requested"

    for opt in ("Emirates", "British Airways", "Turkish Airlines", "Gulf Air", "flydubai", "PIA", "Saudia"):
        if opt.lower() in full_all_text.lower():
            chosen_option = f"{opt} option discussed on route {meta.enquiry_route or 'enquired route'}"
            break

    for phrase in ("tomorrow morning", "tomorrow afternoon", "tomorrow evening", "later today", "this evening", "6 pm", "5 pm"):
        if phrase in full_all_text.lower():
            callback_time = phrase
            break

    return {
        "outcome": outcome,
        "chosen_option": chosen_option,
        "callback_time": callback_time,
        "notes_for_consultant": (
            f"Caller: {meta.resolved_name() or 'Unknown'}, Route: {meta.enquiry_route or 'N/A'}. "
            f"Termination: {termination_reason}. Turns recorded: {len(transcript)}."
        ),
    }

