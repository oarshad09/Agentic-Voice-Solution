# Ezee Flights — Self-Hosted Outbound Voice Agent (Zaya)

A self-hosted outbound voice agent built on **Python 3.11+** and **LiveKit Agents 1.x (`livekit-agents`)** that executes the workflow defined in [`spec/flight_booking.json`](spec/flight_booking.json).

`spec/flight_booking.json` is loaded **at runtime** as the single source of truth for all prompts, models, voice settings, STT keyterms, turn-taking parameters, tool instructions, and static flight inventory. Known inconsistencies in the source JSON are patched non-destructively at runtime via [`prompt_overrides.yaml`](prompt_overrides.yaml).

---

## Architecture & Node Mapping (`spec/flight_booking.json` → Code)

| JSON Node | Implementation |
| :--- | :--- |
| `Start.masterInstruction` | Global system-prompt prefix (identity & hard rules), always placed first in `AgentSpec.build_system_prompt()` |
| `eandAI.instructions` | Main agent instructions, appended immediately after `masterInstruction` |
| `Start.inputs` / `workflow_inputs` (`caller_name`) | Parsed from LiveKit job/room metadata via `CallMetadata` (`caller_name`, `customer_name`, `enquiry_route`, `phone_number`) |
| `{{$node("Start").caller_name}}`, `{{customer_name}}`, `{{enquiry_route}}` | Rendered before the session starts in `spec_loader.render_placeholders()`. Empty values resolve to `""`, never a literal placeholder |
| `Breez Ears` | `providers/stt.py` (`build_stt`): Deepgram `nova-3` (`language="multi"`) by default, or swappable to `soniox` (`STT_PROVIDER=soniox`). Extracts 76 domain keyterms from `Breez Ears.context`, language hints `["en", "ar", "hi", "ur", "es"]`, and `max_endpoint_delay_ms=1720` |
| `OpenAI Brain` | `openai.LLM(model="gpt-5.4-mini", max_completion_tokens=500, reasoning_effort="none")` |
| `ElevenLabs Voice` | `elevenlabs.TTS(model="eleven_flash_v2_5")` with `stability=0.4`, `similarity_boost=0.8`, `style=0.35`, `use_speaker_boost=True`, `speed=1.0`, and voice ID from `ELEVEN_VOICE_ID` (defaulting to `ack0QsRaQyDLnVyMQTSd`) |
| `BGNoise_Office` | `BackgroundAudioPlayer` playing `BuiltinAudioClip.OFFICE_AMBIENCE` at `20%` (`0.20`) volume |
| Turn-taking (`minInterruptionWords=3`, `maxEndpointingDelay=2.3`, STT `max_endpoint_delay_ms=1720`) | `AgentSession(min_interruption_words=3, max_endpointing_delay=2.3, vad=silero.VAD.load(), turn_detection=MultilingualModel())` |
| `fillers_enabled` & `naturalness_fillers` | Loaded from `Start` (`["yeah", "honestly", "right", "sure", "i mean"]`); short filler before tool calls when `TOOL_FILLER_AUDIO=true` (in addition to the LLM's natural bridge line) |
| `SearchFlights` | `@function_tool` `search_flights(query: str)` using `SearchFlights.parameters.instructions` verbatim |
| `KBFlight` → `RESPFlight` | `providers/flights.py`: `FlightProvider` interface with `StaticProvider` (v1: raw text) and `KBProvider` (v2: local vector store + query decomposition + cross-feature reranker, `topK=5`) |
| `maxCallDurationMinutes=10`, `silenceTimeoutSeconds=30` | Background watchdog task in `agent.py` that politely closes and hangs up when either the 10-minute call cap or 30-second silence timeout is reached |
| `post_call_model` (`gpt-5.4-mini`) | `post_call.py`: Sends the call transcript to OpenAI on shutdown and writes `calls/<call_id>.json` (`outcome`, `chosen_option`, `callback_time`, `notes_for_consultant`, metadata, and transcript) |

---

## Prompt Fixes Applied via `prompt_overrides.yaml`

The source file `spec/flight_booking.json` is **never modified**. Instead, [`prompt_overrides.yaml`](prompt_overrides.yaml) applies four targeted fixes at runtime in `spec_loader.py`:

1. **Persona Name Standardisation (`Anna` / `Zaya` / `Nour` → `Zaya`)**
   - *Issue in JSON*: `Start.parameters.masterInstruction` says `"You are only Anna"`, `eandAI.parameters.instructions` says `"You are Zaya"`, and `Start.parameters.inputs[0].description` says `"leave empty and Nour will simply ask for it."`
   - *Override*: Standardises all three to **Zaya** in `persona_standardization.replacements`.
2. **Input Variable Mapping (`{{customer_name}}` and `{{enquiry_route}}`)**
   - *Issue in JSON*: `eandAI.parameters.instructions` references `{{customer_name}}` and `{{enquiry_route}}` (as well as `{{$node("Start").caller_name}}` and `{{$phoneNumber('international', 'external_number')}}`), while `workflow_inputs` only declares `caller_name`.
   - *Override*: Maps `customer_name` ↔ `caller_name` bidirectionally and adds optional dispatch inputs `enquiry_route` and `phone_number`. Any missing value renders as `""`, never a literal `{{...}}` string.
3. **`SearchFlights` Input Query Description (Real-Estate Examples → Flight Examples)**
   - *Issue in JSON*: `SearchFlights.parameters.inputs[0].description` contains copy-pasted real-estate examples (`'available sea-view two-beds in Coral around 3.5'`, `'Pearl payment schedule'`, `'schools near Dubai Islands'`).
   - *Override*: Replaces the `query` parameter description with flight-specific examples (`'return Dubai to London Heathrow 12 Dec to 2 Jan, 2 adults 1 child, direct'`, `'baggage and refund rules on the Emirates morning option'`, `'cheapest return flights Dubai to Karachi or Lahore in December'`).
4. **STT Language Hints (`en`, `ar`, `hi` → `en`, `ar`, `hi`, `ur`, `es`)**
   - *Issue in JSON*: `Start.masterInstruction` and `eandAI.instructions` promise the agent will mirror English, Arabic, Hindi, **Urdu**, and **Spanish**, but `Breez Ears.parameters.language_hints` only lists `["en", "ar", "hi"]`.
   - *Override*: Appends `"ur"` and `"es"` to `Breez Ears.language_hints`.

---

## Setup & Installation

### 1. Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required keys for live voice calls:
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY` (or `SONIOX_API_KEY` if `STT_PROVIDER=soniox`)
- `ELEVEN_API_KEY` (and optional `ELEVEN_VOICE_ID`, defaults to `ack0QsRaQyDLnVyMQTSd`)
- `SIP_OUTBOUND_TRUNK_ID` (for outbound telephony via Twilio or a UAE SIP carrier)

### 2. Install Dependencies (Python 3.11+)

```bash
pip install -r requirements.txt
python agent.py download-files
```

---

## Running the Agent

### 0. Localhost Frontend Web App (`http://localhost:8080`)

Launch the interactive browser console to talk to Zaya via your microphone or text, inspect `search_flights` vector chunks, trigger outbound SIP calls, and browse saved `calls/*.json` summaries:

```bash
python web_app.py --port 8080
```

Then open **http://localhost:8080** in your browser.

### 1. Console Mode (Laptop Mic & Speakers — No SIP Required)

Talk to Zaya directly from your terminal in English, Arabic, Urdu, Hindi, or Spanish:

```bash
python agent.py console
```

You can customize the caller context for console mode via `.env` or inline environment variables:

```bash
DEFAULT_CALLER_NAME="Omair" DEFAULT_ENQUIRY_ROUTE="Dubai to London" python agent.py console
```

When you end the console session (`Ctrl+C` or `[Q]`), a post-call summary is automatically written to `calls/<call_id>.json`.

### 2. Outbound SIP Call Worker + Dispatch CLI

Start the LiveKit worker in one terminal:

```bash
python agent.py dev
```

In a second terminal, place an outbound call to your mobile number:

```bash
python dispatch_call.py --to +9715XXXXXXX --name "Omair" --route "Dubai to London"
```

To validate the dispatch payload and rendered prompt without dialing a live trunk:

```bash
python dispatch_call.py --to +9715XXXXXXX --name "Omair" --route "Dubai to London" --dry-run
```

### 3. Docker Deployment

```bash
docker build -t ezee-flights-agent .
docker run --env-file .env -v $(pwd)/calls:/app/calls ezee-flights-agent
```

---

## Testing (5 Text-Only Conversations + Tool Execution)

Run [`test_conversations.py`](test_conversations.py) to execute all 5 required conversation scenarios against the prompt and `search_flights` tool without audio:

1. **Happy path booking** (Dubai to London family return, chooses Emirates direct morning option, connects to consultant)
2. **Too-expensive objection** (2,400 AED is over budget; agent uses levers: 1 day earlier on 11 Dec + 1 stop via Istanbul / Bahrain; customer picks Turkish Airlines 11 Dec)
3. **Visa question** (Pakistani passport holder asks about UK visa requirements; agent refuses immigration advice per hard rules and directs to embassy/airline)
4. **Refund on an existing booking** (Caller asks about a delayed refund; agent does not promise amount/timeline, captures PNR + lead passenger name, escalates to support)
5. **"Don't call me" opt-out** (Caller asks to be removed from calling list; agent apologises once, confirms removal, and ends immediately)

```bash
python test_conversations.py
```

Each scenario prints its full transcript (including `search_flights` tool invocations) and writes its post-call summary JSON to `calls/test_<n>_*.json`.

