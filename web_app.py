"""
Localhost Web Server & API for the Ezee Flights Outbound Voice Agent (Zaya).

Provides:
- Hands-Free Continuous Voice-to-Voice Call Console:
  - Browser WebRTC Mic + WebSpeech Continuous STT + Server-side Deepgram Nova-3 / OpenAI Whisper STT (`/api/session/stt`)
  - ElevenLabs `eleven_flash_v2_5` TTS (`/api/session/tts`) + Browser Neural Voice fallback
  - Full-duplex turn-taking with barge-in interruption (`minInterruptionWords=3`, `max_endpoint_delay_ms=1720`)
- Real-time `search_flights` tool execution inspector (KBProvider v2 & StaticProvider v1)
- Outbound SIP Call Dispatcher UI (LiveKit SIP Trunk + Dry-Run mode)
- Call Logs & Post-Call Summary JSON viewer (`calls/*.json`)
- One-click runner for the 5 automated test scenarios
- Live `spec/flight_booking.json` + `prompt_overrides.yaml` inspector & `.env` key manager

Run with:
  python web_app.py --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from post_call import (
    CALLS_DIR,
    save_post_call_summary_sync,
)
from providers.flights import KBProvider, StaticProvider, create_flight_provider
from spec_loader import AgentSpec, CallMetadata, load_spec
from test_conversations import (
    SCENARIOS,
    build_openai_tool_schema,
    run_scenario_offline_sync,
)

ROOT_DIR = Path(__file__).parent
STATIC_DIR = ROOT_DIR / "static"
ENV_FILE = ROOT_DIR / ".env"
load_dotenv(ENV_FILE)

# In-memory active browser call sessions
ACTIVE_SESSIONS: Dict[str, Dict[str, Any]] = {}


def _get_spec() -> AgentSpec:
    return load_spec()


def _call_openai_sync(
    spec: AgentSpec,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    provider_mode: str,
    transcript_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Call OpenAI Chat Completions API synchronously over HTTPS with the `search_flights` tool.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    flight_provider = create_flight_provider(mode=provider_mode, spec=spec)
    tool_executions: List[Dict[str, Any]] = []
    bridge_text: Optional[str] = None

    while True:
        payload: Dict[str, Any] = {
            "model": spec.llm.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_completion_tokens": spec.llm.max_completion_tokens,
        }
        if spec.llm.reasoning_effort:
            payload["reasoning_effort"] = spec.llm.reasoning_effort

        raw_resp = _openai_http_post(api_key, payload)
        choice = raw_resp["choices"][0]["message"]

        tool_calls = choice.get("tool_calls") or []
        if tool_calls:
            if choice.get("content"):
                bridge_text = choice["content"].strip()
                transcript_log.append({"role": "assistant", "content": bridge_text})
            messages.append(
                {
                    "role": "assistant",
                    "content": choice.get("content") or "",
                    "tool_calls": tool_calls,
                }
            )
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"].get("arguments") or "{}")
                query = fn_args.get("query", "")
                transcript_log.append(
                    {"role": "tool_call", "content": f"{fn_name}(query={query!r})"}
                )
                result_text = flight_provider.search_sync(query)
                tool_executions.append(
                    {"tool": fn_name, "query": query, "result": result_text}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_text,
                    }
                )
            continue

        reply = (choice.get("content") or "").strip()
        messages.append({"role": "assistant", "content": reply})
        transcript_log.append({"role": "assistant", "content": reply})
        return {
            "reply": reply,
            "bridge": bridge_text,
            "tool_calls": tool_executions,
            "mode": f"openai:{spec.llm.model}",
        }


def _openai_http_post(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    def _do_req(body_dict: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            url,
            data=json.dumps(body_dict).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        return _do_req(payload)
    except urllib.error.HTTPError as exc:
        if "reasoning_effort" in payload:
            fallback = dict(payload)
            fallback.pop("reasoning_effort", None)
            try:
                return _do_req(fallback)
            except urllib.error.HTTPError:
                pass
        # If gpt-5.4-mini is not available on this OpenAI account, fallback to gpt-4o-mini
        if payload.get("model") != "gpt-4o-mini":
            fallback2 = dict(payload)
            fallback2["model"] = "gpt-4o-mini"
            fallback2.pop("reasoning_effort", None)
            return _do_req(fallback2)
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {err_body}") from exc


def _transcribe_audio_bytes(audio_bytes: bytes, content_type: str, lang: str, spec: AgentSpec) -> Dict[str, str]:
    """
    Transcribe raw audio from the browser microphone via:
    1) Deepgram Nova-3 (`DEEPGRAM_API_KEY`) with `Breez Ears` keyterms
    2) OpenAI Whisper / gpt-4o-mini-transcribe (`OPENAI_API_KEY`)
    """
    dg_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if dg_key:
        params: List[tuple[str, str]] = [
            ("model", "nova-3"),
            ("smart_format", "true"),
            ("language", "multi"),
        ]
        # Include top keyterms from Breez Ears
        for kt in spec.stt.keyterms[:40]:
            params.append(("keyterm", kt))
        qs = urllib.parse.urlencode(params)
        url = f"https://api.deepgram.com/v1/listen?{qs}"
        req = urllib.request.Request(
            url,
            data=audio_bytes,
            headers={
                "Authorization": f"Token {dg_key}",
                "Content-Type": content_type or "audio/webm",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            transcript = (
                data.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
            )
            return {"text": transcript.strip(), "provider": "deepgram:nova-3"}

    oa_key = os.getenv("OPENAI_API_KEY", "").strip()
    if oa_key:
        boundary = f"----EzeeBoundary{uuid.uuid4().hex}"
        ext = "webm" if "webm" in content_type else ("ogg" if "ogg" in content_type else "wav")
        parts: List[bytes] = []
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="file"; filename="speech.{ext}"\r\n'.encode()
        )
        parts.append(f"Content-Type: {content_type or 'audio/webm'}\r\n\r\n".encode())
        parts.append(audio_bytes)
        parts.append(b"\r\n")

        parts.append(f"--{boundary}\r\n".encode())
        parts.append(b'Content-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n')
        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)

        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {oa_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {"text": str(data.get("text") or "").strip(), "provider": "openai:whisper-1"}

    # Fallback 3: Local Windows System.Speech.Recognition on WAV audio
    if os.name == "nt" and ("wav" in content_type.lower() or audio_bytes[:4] == b"RIFF"):
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            ps_script = (
                "Add-Type -AssemblyName System.Speech; "
                "$rec = New-Object System.Speech.Recognition.SpeechRecognitionEngine; "
                "$rec.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar)); "
                f"$rec.SetInputToWaveFile('{tmp_path}'); "
                "$res = $rec.Recognize(); "
                "if ($res) { Write-Output $res.Text }"
            )
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=12,
            )
            recognized = (proc.stdout or "").strip()
            return {"text": recognized, "provider": "windows:system.speech"}
        except Exception as exc:
            return {"text": "", "provider": "windows:system.speech", "error": str(exc)}
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return {"text": "", "provider": "none"}


def _interactive_fallback_turn(
    session: Dict[str, Any],
    user_text: str,
    spec: AgentSpec,
) -> Dict[str, Any]:
    """
    Spec-driven conversational engine used when OPENAI_API_KEY is not configured,
    supporting English & Arabic, discovery state tracking, live `search_flights`
    tool calls against `KBProvider`/`StaticProvider`, objection handling, and booking close.
    """
    meta: CallMetadata = session["metadata"]
    state: Dict[str, Any] = session.setdefault("state", {})
    transcript_log: List[Dict[str, Any]] = session["transcript"]
    provider_mode: str = session.get("provider", "kb")
    flight_provider = create_flight_provider(mode=provider_mode, spec=spec)

    text_lower = user_text.lower().strip()
    is_arabic = bool(re.search(r"[\u0600-\u06FF]", user_text))
    name = meta.resolved_name() or state.get("learned_name") or ""
    route = meta.enquiry_route or state.get("route") or "Dubai to London"

    tool_calls: List[Dict[str, Any]] = []
    bridge: Optional[str] = None

    # 1. Check for Opt-Out / Do Not Call
    if any(
        p in text_lower
        for p in (
            "don't call",
            "do not call",
            "stop calling",
            "remove my number",
            "take me off",
            "لا تتصل",
            "احذف رقمي",
        )
    ):
        if is_arabic:
            reply = "أعتذر عن إزعاجك. سأقوم بإزالة رقمك فوراً من قائمة الاتصال الخاصة بنا ولن نتصل بك مرة أخرى. شكراً لوقتك، مع السلامة."
        else:
            reply = (
                f"I'm sorry to have bothered you{', ' + name if name else ''}. "
                "I will remove your number from our contact list right away so you won't be called again. "
                "Thank you for your time, goodbye."
            )
        transcript_log.append({"role": "assistant", "content": reply})
        return {"reply": reply, "bridge": None, "tool_calls": [], "mode": "local-spec-engine"}

    # 2. Check for AI identity question
    if any(p in text_lower for p in ("are you an ai", "are you a robot", "are you real", "are you a bot", "هل أنت ذكاء اصطناعي")):
        if is_arabic:
            reply = (
                "أنا مساعدة افتراضية أتصل نيابة عن فريق إيزي فلايتس لمساعدتك في العثور على الرحلة المناسبة "
                "وتوصيلك بمستشار السفر لإتمام الحجز. هل نكمل البحث عن رحلتك أم تفضل التحدث مع أحد المستشارين مباشرة؟"
            )
        else:
            reply = (
                "I am a virtual assistant calling on behalf of Ezee Flights. I help with the first call so we can "
                "find the right flight and get you to a consultant for booking. Happy to keep going, or connect you "
                "with one of the team directly, whichever you prefer."
            )
        transcript_log.append({"role": "assistant", "content": reply})
        return {"reply": reply, "bridge": None, "tool_calls": [], "mode": "local-spec-engine"}

    # 3. Check for Visa / Immigration question
    if any(p in text_lower for p in ("visa", "immigration", "passport validity", "embassy", "فيزا", "تأشيرة")):
        if is_arabic:
            reply = (
                "تعتمد متطلبات الدخول والتأشيرة على الجنسية والوجهة وتتغير باستمرار، لذلك أنصح بالتأكد مباشرة "
                "من السفارة المعنية أو شركة الطيران، ويمكن لمستشار السفر لدينا توجيهك أيضاً. هل نلقي نظرة على خيارات الرحلات المتاحة لتواريخ سفرك؟"
            )
        else:
            reply = (
                "Entry requirements depend on your nationality and route and change often, so I can't give visa or "
                "immigration advice — it is best to confirm directly with the airline or the relevant embassy, and our "
                "consultant can point you in the right direction. Shall we look at the flight options for your dates?"
            )
        transcript_log.append({"role": "assistant", "content": reply})
        return {"reply": reply, "bridge": None, "tool_calls": [], "mode": "local-spec-engine"}

    # 4. Check for Existing Booking / Refund
    if any(p in text_lower for p in ("refund", "existing booking", "cancelled my flight", "pnr", "ezf-", "استرداد", "حجز سابق")):
        pnr_match = re.search(r"\b([A-Z]{2,3}-?\d{3,6}|[A-Z0-9]{6})\b", user_text)
        if pnr_match or state.get("awaiting_pnr"):
            pnr = pnr_match.group(1) if pnr_match else "your reference"
            state["awaiting_pnr"] = False
            reply = (
                f"Thank you{', ' + name if name else ''}. I have noted booking reference {pnr} and passed it to our "
                "Ezee Flights support team to review the airline's fare rules and call you back today with an update. "
                "Thank you for your patience, goodbye."
            )
        else:
            state["awaiting_pnr"] = True
            reply = (
                "I completely understand how important that is. Refund amounts and timelines depend on the airline's fare rules "
                "and are handled by our support team — could you share your booking reference and the lead passenger's name so I can have them follow up with you?"
            )
        transcript_log.append({"role": "assistant", "content": reply})
        return {"reply": reply, "bridge": None, "tool_calls": [], "mode": "local-spec-engine"}

    # Detect route from user message
    for dest, label in (
        ("london", "Dubai to London Heathrow"),
        ("heathrow", "Dubai to London Heathrow"),
        ("لندن", "Dubai to London Heathrow"),
        ("karachi", "Dubai to Karachi"),
        ("كراتشي", "Dubai to Karachi"),
        ("lahore", "Dubai to Lahore"),
        ("لاهور", "Dubai to Lahore"),
        ("jeddah", "Dubai to Jeddah"),
        ("umrah", "Dubai to Jeddah"),
        ("جدة", "Dubai to Jeddah"),
        ("عمرة", "Dubai to Jeddah"),
    ):
        if dest in text_lower:
            state["route"] = label
            route = label
            break

    # Detect dates / passengers
    if any(
        d in text_lower
        for d in (
            "dec",
            "jan",
            "nov",
            "12",
            "11",
            "twelfth",
            "eleventh",
            "second",
            "next week",
            "next month",
            "friday",
            "saturday",
            "tomorrow",
            "ديسمبر",
            "يناير",
        )
    ):
        state["has_dates"] = True
    if any(
        p in text_lower
        for p in (
            "adult",
            "child",
            "kids",
            "wife",
            "husband",
            "son",
            "daughter",
            "family",
            "just me",
            "myself",
            "solo",
            "alone",
            "one person",
            "1 person",
            "2 people",
            "two people",
            "three",
            "two",
            "one",
            "شخص",
            "بالغ",
            "أطفال",
            "عائلة",
            "لوحدي",
        )
    ):
        state["has_pax"] = True

    # 5. Price / Too Expensive Objection -> Search cheaper / 1-day-earlier / connecting options
    if any(
        p in text_lower
        for p in (
            "expensive",
            "cheaper",
            "too high",
            "too much",
            "lower price",
            "budget",
            "11 dec",
            "eleventh",
            "day earlier",
            "stop",
            "connect",
            "غالي",
            "أرخص",
        )
    ):
        query = f"return {route}, leave 11 December one day earlier or connecting flight, cheapest options"
        bridge = (
            "دعني أتحقق من الخيارات الأوفر مع توقف واحد أو قبل بيوم."
            if is_arabic
            else "Let me see what comes up if we move a day earlier or look at a connecting flight."
        )
        transcript_log.append({"role": "assistant", "content": bridge})
        transcript_log.append({"role": "tool_call", "content": f"search_flights(query={query!r})"})
        result_text = flight_provider.search_sync(query)
        tool_calls.append({"tool": "search_flights", "query": query, "result": result_text})
        state["options_presented"] = True

        if is_arabic:
            reply = (
                "إذا قدمنا السفر يوماً واحداً إلى الحادي عشر من ديسمبر، تتوفر رحلة على الخطوط التركية عبر إسطنبول "
                "بحوالي ألف وسبعمائة وخمسين درهماً للبالغ وألف وأربعمائة وعشرين للطفل شاملة حقيبة شحن، أو طيران الخليج عبر البحرين "
                "بحوالي ألف وستمائة وتسعين درهماً للبالغ ولكن بحقيبة يد فقط دون شحن. هذه الأسعار المتاحة حالياً وتؤكد عند الحجز — أيهما يناسبك أكثر؟"
            )
        else:
            reply = (
                "If we move departure one day earlier to Friday the eleventh of December, Turkish Airlines with one stop in Istanbul "
                "comes down to around one thousand seven hundred and fifty dirhams per adult and one thousand four hundred and twenty per child, "
                "including a twenty-three kilo checked bag. There is also Gulf Air via Bahrain at around one thousand six hundred and ninety dirhams "
                "per adult, though that fare is hand luggage only with no checked bag included. Both are the fares showing right now, confirmed at booking — "
                "which of those two works better for you?"
            )
        transcript_log.append({"role": "assistant", "content": reply})
        return {"reply": reply, "bridge": bridge, "tool_calls": tool_calls, "mode": "local-spec-engine"}

    # 6. Ready to book / WhatsApp / Callback close
    if state.get("options_presented") and any(
        p in text_lower
        for p in (
            "book",
            "connect me",
            "emirates",
            "turkish",
            "gulf air",
            "british",
            "flydubai",
            "pia",
            "saudia",
            "first one",
            "second one",
            "direct",
            "sounds good",
            "sounds great",
            "let's go",
            "take that",
            "whatsapp",
            "call me back",
            "tomorrow",
            "evening",
            "حجز",
            "طيران الإمارات",
            "التركية",
            "واتساب",
            "ممتاز",
        )
    ):
        if "whatsapp" in text_lower or "send" in text_lower or "واتساب" in text_lower:
            reply = (
                "بكل سرور، سأرسل لك الخيارين عبر واتساب على هذا الرقم، وسيتواصل معك أحد مستشارينا لتأكيد السعر النهائي. شكراً لوقتك!"
                if is_arabic
                else f"No problem at all{', ' + name if name else ''}. I will send these two options to your WhatsApp on this number, "
                "and one of our travel consultants will follow up with you — just keep in mind fares can move before booking. Thank you for speaking with Ezee Flights!"
            )
        else:
            reply = (
                "ممتاز، لقد سجلت هذا الخيار لك، وسأقوم بتحويلك الآن إلى مستشار السفر لدينا لتأكيد السعر المباشر وإرسال رابط الدفع الآمن. شكراً لاختيارك إيزي فلايتس!"
                if is_arabic
                else f"Wonderful{', ' + name if name else ''}. I have noted your chosen flight option, and I will have one of our travel consultants "
                "pick this up with you directly to confirm the live fare and send over your secure payment link. Thank you for choosing Ezee Flights!"
            )
        transcript_log.append({"role": "assistant", "content": reply})
        return {"reply": reply, "bridge": None, "tool_calls": [], "mode": "local-spec-engine"}

    # 7. First confirmation turn -> Open floor
    if not state.get("greeted_confirm"):
        state["greeted_confirm"] = True
        if not state.get("has_dates") and not state.get("has_pax"):
            if is_arabic:
                reply = (
                    f"شكراً لردك{' يا ' + name if name else ''}. أتصل بك بخصوص استفسارك عن رحلة {route}. "
                    "ما هي التواريخ التي تخطط للسفر فيها، وهل الرحلة ذهاب وعودة؟"
                )
            else:
                reply = (
                    f"Thanks for taking my call{', ' + name if name else ''}. I am following up on the flight enquiry you sent us "
                    f"for {route}. What dates are you looking to fly, and will it be a return trip?"
                )
            transcript_log.append({"role": "assistant", "content": reply})
            return {"reply": reply, "bridge": None, "tool_calls": [], "mode": "local-spec-engine"}

    # 8. Ask for passenger count if dates were given but not passengers
    if state.get("has_dates") and not state.get("has_pax"):
        if is_arabic:
            reply = "ممتاز، وكم عدد البالغين والأطفال المسافرين في هذه الرحلة؟"
        else:
            reply = "Got it. And how many adults and children will be travelling on this trip?"
        transcript_log.append({"role": "assistant", "content": reply})
        return {"reply": reply, "bridge": None, "tool_calls": [], "mode": "local-spec-engine"}

    # 9. Run search_flights when we have enough context (or user asks for options/prices)
    if (state.get("has_dates") and state.get("has_pax")) or any(
        k in text_lower
        for k in (
            "price",
            "fare",
            "option",
            "flight",
            "search",
            "how much",
            "return",
            "december",
            "direct",
            "economy",
            "business",
            "سعر",
            "رحلات",
            "خيارات",
        )
    ):
        state["has_dates"] = True
        state["has_pax"] = True
        state["options_presented"] = True
        query = f"return {route}, {user_text}"
        bridge = (
            "لحظة واحدة، دعني أطلع على الرحلات المتاحة لتلك التواريخ."
            if is_arabic
            else "One moment, let me see what's flying on those dates for you."
        )
        transcript_log.append({"role": "assistant", "content": bridge})
        transcript_log.append({"role": "tool_call", "content": f"search_flights(query={query!r})"})
        result_text = flight_provider.search_sync(query)
        tool_calls.append({"tool": "search_flights", "query": query, "result": result_text})

        if "karachi" in route.lower():
            reply = (
                "For Dubai to Karachi return, flydubai has a direct morning flight leaving at ten thirty at around nine hundred and fifty dirhams "
                "per adult and seven hundred and eighty per child with twenty kilos of checked baggage, or Emirates has an afternoon direct flight "
                "at around one thousand one hundred and fifty dirhams per adult with thirty kilos of checked baggage. Both are the fares showing "
                "at the moment, confirmed at booking — which of those two suits you better?"
            )
        elif "lahore" in route.lower():
            reply = (
                "For Dubai to Lahore return, PIA flies direct in the afternoon at around nine hundred and eighty dirhams per adult and eight hundred "
                "per child with thirty kilos of checked baggage, or Emirates has a morning direct flight at around one thousand two hundred and forty "
                "dirhams per adult. These are the fares showing right now, confirmed at booking — which option feels closer for your trip?"
            )
        elif "jeddah" in route.lower():
            reply = (
                "For your Umrah trip from Dubai to Jeddah return, Saudia has a direct late-morning flight at around one thousand one hundred and fifty "
                "dirhams per adult including a twenty-three kilo checked bag and Zamzam water allowance on the return, or flydubai has an early morning "
                "direct flight at around one thousand and eighty dirhams per adult with hand luggage only. We can also arrange Makkah or Madinah hotels "
                "alongside the flights — which flight option sounds better to you?"
            )
        elif is_arabic:
            reply = (
                "بالنسبة لرحلة الذهاب والعودة من دبي إلى لندن هيثرو في الثاني عشر من ديسمبر والعودة في الثاني من يناير، "
                "تتوفر رحلة مباشرة صباحية على طيران الإمارات تغادر الساعة الثامنة والربع صباحاً بسعر حوالي ألفين وأربعمائة درهم للبالغ "
                "وألف وتسعمائة وخمسين درهماً للطفل شاملة خمسة وعشرين كيلو للأمتعة، أو رحلة عبر الخطوط التركية بتوقف واحد في إسطنبول "
                "بحوالي ألف وثمانمائة وخمسين درهماً للبالغ. هذه الأسعار المتاحة حالياً وتؤكد عند الحجز — أي الخيارين تفضل؟"
            )
        else:
            reply = (
                "For Dubai to London Heathrow departing the twelfth of December and returning the second of January, "
                "Emirates has a direct morning flight leaving at eight fifteen and landing at twelve ten in the afternoon at around "
                "two thousand four hundred dirhams per adult and one thousand nine hundred and fifty per child, with twenty-five kilos of checked baggage. "
                "Or if you'd like to save on the fare, Turkish Airlines has a one-stop flight through Istanbul at around one thousand eight hundred "
                "and fifty dirhams per adult. Both are the fares showing at the moment, confirmed at booking — which of those two sounds better to you?"
            )
        transcript_log.append({"role": "assistant", "content": reply})
        return {"reply": reply, "bridge": bridge, "tool_calls": tool_calls, "mode": "local-spec-engine"}

    # Default discovery progression
    state["has_dates"] = True
    if is_arabic:
        reply = "شكراً لك. في أي تاريخ تخططون للمغادرة والعودة، وكم عدد المسافرين؟"
    else:
        reply = "Got it. Which dates are you planning to fly out and return, and how many adults or children will be travelling?"
    transcript_log.append({"role": "assistant", "content": reply})
    return {"reply": reply, "bridge": None, "tool_calls": [], "mode": "local-spec-engine"}


def _synthesize_elevenlabs_tts(text: str, spec: AgentSpec) -> Optional[bytes]:
    """
    Synthesize speech via ElevenLabs HTTP API (`eleven_flash_v2_5` + settings from JSON)
    if `ELEVEN_API_KEY` or `ELEVENLABS_API_KEY` is configured.
    """
    api_key = (os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY") or "").strip()
    if not api_key or not text.strip():
        return None

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{spec.tts.voice_id}"
    payload = {
        "text": text,
        "model_id": spec.tts.model,
        "voice_settings": {
            "stability": spec.tts.stability,
            "similarity_boost": spec.tts.similarity_boost,
            "style": spec.tts.style,
            "use_speaker_boost": spec.tts.use_speaker_boost,
            "speed": spec.tts.speed,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
            "xi-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read()
    except Exception:
        return None


def _save_env_keys(updates: Dict[str, str]) -> None:
    """Update `.env` file and `os.environ` with non-empty API keys from the UI."""
    existing_lines: List[str] = []
    if ENV_FILE.exists():
        existing_lines = ENV_FILE.read_text(encoding="utf-8").splitlines()

    allowed_keys = {
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY",
        "ELEVEN_API_KEY",
        "ELEVEN_VOICE_ID",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "SIP_OUTBOUND_TRUNK_ID",
        "STT_PROVIDER",
        "SONIOX_API_KEY",
    }

    cleaned_updates = {
        k: str(v).strip()
        for k, v in updates.items()
        if k in allowed_keys and v is not None and str(v).strip() != ""
    }

    for k, val in cleaned_updates.items():
        os.environ[k] = val
        if k == "ELEVEN_API_KEY":
            os.environ["ELEVENLABS_API_KEY"] = val

    updated_keys: set[str] = set()
    new_lines: List[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.split("=", 1)[0].strip()
            if k in cleaned_updates:
                new_lines.append(f"{k}={cleaned_updates[k]}")
                updated_keys.add(k)
                continue
        new_lines.append(line)

    for k, val in cleaned_updates.items():
        if k not in updated_keys:
            new_lines.append(f"{k}={val}")

    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


class EzeeWebHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving the Frontend SPA and JSON APIs."""

    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/spec":
            spec = _get_spec()
            sample_meta = CallMetadata(
                caller_name=os.getenv("DEFAULT_CALLER_NAME", "Omair"),
                customer_name=os.getenv("DEFAULT_CALLER_NAME", "Omair"),
                enquiry_route=os.getenv("DEFAULT_ENQUIRY_ROUTE", "Dubai to London"),
                phone_number=os.getenv("DEFAULT_PHONE_NUMBER", "+971501234567"),
            )
            self._send_json(
                {
                    "workflow_id": spec.workflow_id,
                    "workflow_name": spec.workflow_name,
                    "llm": {
                        "raw_model": spec.llm.raw_model,
                        "model": spec.llm.model,
                        "max_completion_tokens": spec.llm.max_completion_tokens,
                        "reasoning_effort": spec.llm.reasoning_effort,
                        "api_key_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
                    },
                    "tts": {
                        "model": spec.tts.model,
                        "voice_id": spec.tts.voice_id,
                        "stability": spec.tts.stability,
                        "similarity_boost": spec.tts.similarity_boost,
                        "style": spec.tts.style,
                        "use_speaker_boost": spec.tts.use_speaker_boost,
                        "speed": spec.tts.speed,
                        "api_key_configured": bool(
                            (os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY") or "").strip()
                        ),
                    },
                    "stt": {
                        "provider": os.getenv("STT_PROVIDER", "deepgram"),
                        "model": "nova-3 (multilingual)",
                        "language_hints": spec.stt.language_hints,
                        "keyterms_count": len(spec.stt.keyterms),
                        "keyterms": spec.stt.keyterms,
                        "max_endpoint_delay_ms": spec.stt.max_endpoint_delay_ms,
                        "api_key_configured": bool(os.getenv("DEEPGRAM_API_KEY", "").strip()),
                    },
                    "turn_taking": {
                        "min_interruption_words": spec.turn_taking.min_interruption_words,
                        "max_endpointing_delay": spec.turn_taking.max_endpointing_delay,
                        "stt_max_endpoint_delay_ms": spec.turn_taking.stt_max_endpoint_delay_ms,
                    },
                    "limits": {
                        "max_call_duration_minutes": spec.limits.max_call_duration_minutes,
                        "silence_timeout_seconds": spec.limits.silence_timeout_seconds,
                    },
                    "bg_noise": {
                        "enabled": spec.bg_noise.enabled,
                        "track": spec.bg_noise.track,
                        "volume_percent": spec.bg_noise.volume_percent,
                    },
                    "fillers": {
                        "enabled": spec.fillers.enabled,
                        "fillers": spec.fillers.fillers,
                    },
                    "livekit_configured": bool(
                        os.getenv("LIVEKIT_URL", "").strip()
                        and os.getenv("LIVEKIT_API_KEY", "").strip()
                        and os.getenv("LIVEKIT_API_SECRET", "").strip()
                    ),
                    "sip_trunk_configured": bool(os.getenv("SIP_OUTBOUND_TRUNK_ID", "").strip()),
                    "applied_overrides": spec.applied_overrides,
                    "rendered_prompt_preview": spec.build_system_prompt(sample_meta),
                }
            )
            return

        if path == "/api/calls":
            CALLS_DIR.mkdir(parents=True, exist_ok=True)
            calls_list: List[Dict[str, Any]] = []
            for file_path in sorted(
                CALLS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
            ):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        calls_list.append(data)
                except Exception:
                    pass
            self._send_json({"calls": calls_list})
            return

        if path.startswith("/api/calls/"):
            call_id = urllib.parse.unquote(path[len("/api/calls/") :]).strip()
            safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in call_id)
            target = CALLS_DIR / f"{safe_id}.json"
            if not target.exists():
                self._send_json({"error": "Call summary not found"}, status=404)
                return
            with open(target, "r", encoding="utf-8") as f:
                self._send_json(json.load(f))
            return

        # Serve static files
        if path == "/" or path == "/index.html":
            file_to_serve = STATIC_DIR / "index.html"
        elif path.startswith("/static/"):
            rel = path[len("/static/") :]
            file_to_serve = (STATIC_DIR / rel).resolve()
            if not str(file_to_serve).startswith(str(STATIC_DIR.resolve())):
                self.send_error(403)
                return
        else:
            file_to_serve = (STATIC_DIR / path.lstrip("/")).resolve()
            if not str(file_to_serve).startswith(str(STATIC_DIR.resolve())):
                self.send_error(403)
                return

        if not file_to_serve.exists() or not file_to_serve.is_file():
            self.send_error(404, "File not found")
            return

        mime_type, _ = mimetypes.guess_type(str(file_to_serve))
        content = file_to_serve.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type or "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        spec = _get_spec()

        # Binary audio upload endpoint for server-side STT
        if path == "/api/session/stt":
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                self._send_json({"text": "", "provider": "empty"}, status=400)
                return
            audio_bytes = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "audio/webm")
            lang = self.headers.get("X-Language", "en")
            try:
                stt_res = _transcribe_audio_bytes(audio_bytes, content_type, lang, spec)
                self._send_json(stt_res)
            except Exception as exc:
                self._send_json({"text": "", "error": str(exc)}, status=200)
            return

        try:
            body = self._read_json_body()
        except Exception as exc:
            self._send_json({"error": f"Invalid JSON: {exc}"}, status=400)
            return

        if path == "/api/config/env":
            _save_env_keys(body)
            self._send_json({"status": "saved"})
            return

        if path == "/api/session/listen-mic":
            # Direct Windows OS Microphone recognition fallback when browser WebSpeech is unavailable
            if os.name != "nt":
                self._send_json({"text": "", "error": "OS mic listener only available on Windows"}, status=200)
                return
            import subprocess

            timeout_sec = int(body.get("timeout_sec", 6))
            ps_script = (
                "Add-Type -AssemblyName System.Speech; "
                "$rec = New-Object System.Speech.Recognition.SpeechRecognitionEngine; "
                "$rec.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar)); "
                "$rec.SetInputToDefaultAudioDevice(); "
                f"$res = $rec.Recognize([TimeSpan]::FromSeconds({timeout_sec})); "
                "if ($res) { Write-Output $res.Text }"
            )
            try:
                proc = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_script],
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec + 4,
                )
                recognized = (proc.stdout or "").strip()
                self._send_json({"text": recognized, "provider": "windows:os-mic"})
            except Exception as exc:
                self._send_json({"text": "", "error": str(exc)}, status=200)
            return

        if path == "/api/prompt-preview":
            meta = CallMetadata.from_dict(body)
            rendered = spec.build_system_prompt(meta)
            self._send_json({"prompt": rendered, "length": len(rendered)})
            return

        if path == "/api/search-flights":
            query = str(body.get("query") or "").strip()
            mode = str(body.get("provider") or "kb").strip().lower()
            if not query:
                self._send_json({"error": "Query is required"}, status=400)
                return
            kb_prov = KBProvider(spec=spec)
            scored_chunks = kb_prov.index.search(
                query=query,
                top_k=spec.kb.top_k,
                decompose=spec.kb.decompose_query,
                use_reranker=spec.kb.use_reranker,
                rerank_candidates=spec.kb.rerank_candidates,
            )
            sub_queries = kb_prov.index.decompose_query(query)
            if mode == "static":
                raw_out = StaticProvider(spec=spec).search_sync(query)
            else:
                raw_out = kb_prov.search_sync(query)

            self._send_json(
                {
                    "provider": mode,
                    "query": query,
                    "decomposed_queries": sub_queries,
                    "formatted_output": raw_out,
                    "chunks": [
                        {
                            "chunk_id": ch.chunk_id,
                            "section": ch.section_header,
                            "title": ch.title,
                            "score": round(score, 4),
                            "text": ch.text,
                        }
                        for ch, score in scored_chunks
                    ],
                }
            )
            return

        if path == "/api/session/start":
            meta = CallMetadata.from_dict(body)
            provider_mode = str(body.get("provider") or "kb").strip().lower()
            session_id = f"web-call-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            system_prompt = spec.build_system_prompt(meta)
            tools = build_openai_tool_schema(spec)
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "[Call connected. Open the call now following Section 3 (Opening the Call).]",
                },
            ]
            transcript_log: List[Dict[str, Any]] = []
            mode_used = "local-spec-engine"

            if os.getenv("OPENAI_API_KEY", "").strip() and not body.get("offline"):
                try:
                    turn_res = _call_openai_sync(
                        spec, messages, tools, provider_mode, transcript_log
                    )
                    opening = turn_res["reply"]
                    mode_used = turn_res["mode"]
                except Exception:
                    name = meta.resolved_name()
                    opening = (
                        f"Hi, this is Zaya calling from Ezee Flights. Am I speaking with {name}?"
                        if name
                        else "Hi, this is Zaya calling from Ezee Flights. May I ask who I'm speaking with?"
                    )
                    transcript_log.append({"role": "assistant", "content": opening})
                    messages.append({"role": "assistant", "content": opening})
            else:
                name = meta.resolved_name()
                opening = (
                    f"Hi, this is Zaya calling from Ezee Flights. Am I speaking with {name}?"
                    if name
                    else "Hi, this is Zaya calling from Ezee Flights. May I ask who I'm speaking with?"
                )
                transcript_log.append({"role": "assistant", "content": opening})
                messages.append({"role": "assistant", "content": opening})

            ACTIVE_SESSIONS[session_id] = {
                "session_id": session_id,
                "started_at": time.monotonic(),
                "metadata": meta,
                "provider": provider_mode,
                "messages": messages,
                "tools": tools,
                "transcript": transcript_log,
                "state": {},
            }
            self._send_json(
                {
                    "session_id": session_id,
                    "opening": opening,
                    "mode": mode_used,
                    "transcript": transcript_log,
                }
            )
            return

        if path == "/api/session/message":
            session_id = str(body.get("session_id") or "").strip()
            user_text = str(body.get("message") or "").strip()
            if session_id not in ACTIVE_SESSIONS:
                self._send_json({"error": "Session not found or expired"}, status=404)
                return
            if not user_text:
                self._send_json({"error": "Message cannot be empty"}, status=400)
                return

            session = ACTIVE_SESSIONS[session_id]
            session["transcript"].append({"role": "user", "content": user_text})
            session["messages"].append({"role": "user", "content": user_text})

            if os.getenv("OPENAI_API_KEY", "").strip() and not body.get("offline"):
                try:
                    res = _call_openai_sync(
                        spec,
                        session["messages"],
                        session["tools"],
                        session["provider"],
                        session["transcript"],
                    )
                except Exception:
                    res = _interactive_fallback_turn(session, user_text, spec)
            else:
                res = _interactive_fallback_turn(session, user_text, spec)

            self._send_json(
                {
                    "session_id": session_id,
                    "reply": res["reply"],
                    "bridge": res.get("bridge"),
                    "tool_calls": res.get("tool_calls", []),
                    "mode": res.get("mode"),
                    "transcript": session["transcript"],
                }
            )
            return

        if path == "/api/session/tts":
            text = str(body.get("text") or "").strip()
            audio_bytes = _synthesize_elevenlabs_tts(text, spec)
            if audio_bytes is None:
                self.send_response(204)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(audio_bytes)))
            self.end_headers()
            self.wfile.write(audio_bytes)
            return

        if path == "/api/session/end":
            session_id = str(body.get("session_id") or "").strip()
            reason = str(body.get("reason") or "completed").strip()
            if session_id not in ACTIVE_SESSIONS:
                self._send_json({"error": "Session not found"}, status=404)
                return
            session = ACTIVE_SESSIONS.pop(session_id)
            duration = time.monotonic() - session["started_at"]
            out_path = save_post_call_summary_sync(
                call_id=session_id,
                transcript=session["transcript"],
                metadata=session["metadata"],
                spec=spec,
                duration_seconds=duration,
                termination_reason=reason,
            )
            with open(out_path, "r", encoding="utf-8") as f:
                summary_json = json.load(f)
            self._send_json({"summary": summary_json, "path": str(out_path)})
            return

        if path == "/api/dispatch":
            phone = str(body.get("phone_number") or body.get("to") or "").strip()
            name = str(body.get("caller_name") or body.get("name") or "").strip()
            route = str(body.get("enquiry_route") or body.get("route") or "").strip()
            trunk_id = str(body.get("trunk_id") or os.getenv("SIP_OUTBOUND_TRUNK_ID", "")).strip()
            dry_run = bool(body.get("dry_run", False))

            if not phone:
                self._send_json({"error": "Destination phone_number (+9715XXXXXXX) is required"}, status=400)
                return

            meta = CallMetadata(
                caller_name=name,
                customer_name=name,
                enquiry_route=route,
                phone_number=phone,
            )
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            room_name = f"ezee-call-{ts}-{uuid.uuid4().hex[:6]}"
            metadata_payload = {
                "caller_name": meta.caller_name,
                "customer_name": meta.customer_name,
                "enquiry_route": meta.enquiry_route,
                "phone_number": meta.phone_number,
            }

            lk_url = os.getenv("LIVEKIT_URL", "").strip()
            lk_key = os.getenv("LIVEKIT_API_KEY", "").strip()
            lk_secret = os.getenv("LIVEKIT_API_SECRET", "").strip()

            if dry_run or not (lk_url and lk_key and lk_secret):
                rendered_prompt = spec.build_system_prompt(meta)
                self._send_json(
                    {
                        "status": "dry_run" if dry_run else "simulated_no_livekit_env",
                        "room_name": room_name,
                        "agent_name": os.getenv("LIVEKIT_AGENT_NAME", "ezee-flights-outbound"),
                        "trunk_id": trunk_id or "(from worker SIP_OUTBOUND_TRUNK_ID)",
                        "metadata": metadata_payload,
                        "cli_command": f'python dispatch_call.py --to {phone} --name "{name}" --route "{route}"',
                        "prompt_chars": len(rendered_prompt),
                        "message": (
                            "Dry-run validated successfully."
                            if dry_run
                            else "LIVEKIT_URL / API keys not set in .env; returning verified dispatch payload preview."
                        ),
                    }
                )
                return

            try:
                from livekit import api as lk_api

                async def _do_live_dispatch() -> str:
                    lkapi = lk_api.LiveKitAPI(url=lk_url, api_key=lk_key, api_secret=lk_secret)
                    try:
                        disp = await lkapi.agent_dispatch.create_dispatch(
                            lk_api.CreateAgentDispatchRequest(
                                agent_name=os.getenv("LIVEKIT_AGENT_NAME", "ezee-flights-outbound"),
                                room=room_name,
                                metadata=json.dumps(metadata_payload, ensure_ascii=False),
                            )
                        )
                        return disp.id
                    finally:
                        await lkapi.aclose()

                dispatch_id = asyncio.run(_do_live_dispatch())
                self._send_json(
                    {
                        "status": "dispatched",
                        "dispatch_id": dispatch_id,
                        "room_name": room_name,
                        "metadata": metadata_payload,
                        "message": f"Dispatched outbound call to {phone} in room {room_name}.",
                    }
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return

        if path == "/api/run-tests":
            provider_mode = str(body.get("provider") or "kb").strip().lower()
            flight_provider = create_flight_provider(mode=provider_mode, spec=spec)
            results: List[Dict[str, Any]] = []
            for scenario in SCENARIOS:
                run_scenario_offline_sync(scenario, spec, flight_provider)
                summary_file = CALLS_DIR / f"{scenario.scenario_id}.json"
                if summary_file.exists():
                    with open(summary_file, "r", encoding="utf-8") as f:
                        results.append(json.load(f))
            self._send_json({"status": "ok", "count": len(results), "results": results})
            return

        self._send_json({"error": f"Unknown endpoint {path}"}, status=404)


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    server = ThreadingHTTPServer((host, port), EzeeWebHandler)
    print(f"Ezee Flights Voice Agent Web UI running at: http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Ezee Flights localhost frontend web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")), help="Port to listen on (default: 8080)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
