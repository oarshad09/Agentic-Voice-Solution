"""
Runtime loader for spec/flight_booking.json and prompt_overrides.yaml.

Reads the XAIA workflow JSON directly at runtime without modifying it, applies
the fixes defined in prompt_overrides.yaml, extracts node configurations, and
renders the full system prompt with per-call variables.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_SPEC_PATH = Path(__file__).parent / "spec" / "flight_booking.json"
DEFAULT_OVERRIDES_PATH = Path(__file__).parent / "prompt_overrides.yaml"


@dataclass
class CallMetadata:
    """Per-call input variables passed at dispatch or console startup."""

    caller_name: str = ""
    customer_name: str = ""
    enquiry_route: str = ""
    phone_number: str = ""

    def resolved_name(self) -> str:
        """Resolve caller_name / customer_name interchangeably."""
        name = (self.customer_name or self.caller_name or "").strip()
        return name

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "CallMetadata":
        if not data:
            return cls()
        caller_name = str(data.get("caller_name") or data.get("customer_name") or "").strip()
        customer_name = str(data.get("customer_name") or data.get("caller_name") or "").strip()
        enquiry_route = str(data.get("enquiry_route") or data.get("route") or "").strip()
        phone_number = str(
            data.get("phone_number") or data.get("to") or data.get("external_number") or ""
        ).strip()
        return cls(
            caller_name=caller_name,
            customer_name=customer_name,
            enquiry_route=enquiry_route,
            phone_number=phone_number,
        )

    @classmethod
    def from_json(cls, raw_json: Optional[str]) -> "CallMetadata":
        if not raw_json:
            return cls()
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                return cls.from_dict(parsed)
        except json.JSONDecodeError:
            pass
        return cls()


@dataclass
class STTConfig:
    provider_model: str
    context: str
    keyterms: List[str]
    language_hints: List[str]
    max_endpoint_delay_ms: int
    endpoint_sensitivity: float
    enable_diarization: bool


@dataclass
class LLMConfig:
    raw_model: str
    model: str
    reasoning_effort: str
    max_completion_tokens: int


@dataclass
class TTSConfig:
    model: str
    voice_id: str
    stability: float
    similarity_boost: float
    style: float
    use_speaker_boost: bool
    speed: float


@dataclass
class TurnTakingConfig:
    min_interruption_words: int
    max_endpointing_delay: float
    stt_max_endpoint_delay_ms: int
    speak_first_mode: str


@dataclass
class CallLimitsConfig:
    max_call_duration_enabled: bool
    max_call_duration_minutes: float
    silence_timeout_seconds: float


@dataclass
class BackgroundNoiseConfig:
    enabled: bool
    track: str
    volume_percent: int

    @property
    def volume_float(self) -> float:
        return max(0.0, min(1.0, self.volume_percent / 100.0))


@dataclass
class FillersConfig:
    enabled: bool
    fillers: List[str]
    language: str


@dataclass
class SearchToolConfig:
    name: str
    flow_name: str
    instructions: str
    query_description: str


@dataclass
class KnowledgeBaseConfig:
    knowledge_base_id: str
    top_k: int
    model: str
    use_reranker: bool
    decompose_query: bool
    rerank_candidates: int
    output_mode: str
    score_threshold: float
    raw_inventory: str


@dataclass
class PostCallConfig:
    model: str
    reasoning_effort: str


@dataclass
class AgentSpec:
    workflow_id: str
    workflow_name: str
    master_instruction: str
    agent_instructions: str
    stt: STTConfig
    llm: LLMConfig
    tts: TTSConfig
    turn_taking: TurnTakingConfig
    limits: CallLimitsConfig
    bg_noise: BackgroundNoiseConfig
    fillers: FillersConfig
    search_tool: SearchToolConfig
    kb: KnowledgeBaseConfig
    post_call: PostCallConfig
    applied_overrides: List[str] = field(default_factory=list)

    def build_system_prompt(self, metadata: Optional[CallMetadata] = None) -> str:
        """
        Combine Start.masterInstruction (always first) + eandAI.instructions,
        and fill all template placeholders cleanly.
        If any value is empty, inserts an empty string (never a literal placeholder).
        """
        meta = metadata or CallMetadata()
        combined = f"{self.master_instruction.strip()}\n\n{self.agent_instructions.strip()}"
        return render_placeholders(combined, meta)


def render_placeholders(text: str, metadata: CallMetadata) -> str:
    """
    Replace workflow placeholders with runtime values from CallMetadata.
    Empty values become empty strings — never literal '{{...}}' tokens.
    """
    resolved_name = metadata.resolved_name()
    replacements = {
        '{{$node("Start").caller_name}}': resolved_name,
        "{{$node('Start').caller_name}}": resolved_name,
        "{{customer_name}}": resolved_name,
        "{{caller_name}}": resolved_name,
        "{{enquiry_route}}": metadata.enquiry_route or "",
        "{{$phoneNumber('international', 'external_number')}}": metadata.phone_number or "",
        '{{$phoneNumber("international", "external_number")}}': metadata.phone_number or "",
        "{{phone_number}}": metadata.phone_number or "",
    }
    result = text
    for placeholder, val in replacements.items():
        result = result.replace(placeholder, val)

    # Catch any remaining {{...}} placeholders so literal braces are never spoken
    result = re.sub(r"\{\{[^{}]+\}\}", "", result)
    return result


def extract_keyterms_from_context(context: str) -> List[str]:
    """
    Parse the Breez Ears context string into a deduplicated list of keyterms/keywords
    suitable for Deepgram Nova-3 / Soniox boosting.
    """
    keyterms: List[str] = ["Ezee Flights"]

    # Match sections like "Expect: ...", "Airlines: ...", "Places: ..."
    section_pattern = re.compile(
        r"(?:Expect|Airlines|Places)\s*:\s*([^.]+)\.", re.IGNORECASE
    )
    for match in section_pattern.finditer(context):
        items_str = match.group(1)
        for item in items_str.split(","):
            cleaned = item.strip().strip("'\"")
            if cleaned and cleaned not in keyterms:
                keyterms.append(cleaned)

    return keyterms


def _normalize_openai_model(model_name: str) -> str:
    """
    Normalize XAIA model identifiers (e.g. 'gpt-5.4-mini-priority' -> 'gpt-5.4-mini').
    """
    cleaned = model_name.strip()
    if cleaned.endswith("-priority"):
        cleaned = cleaned[: -len("-priority")]
    return cleaned


def load_spec(
    spec_path: Optional[Path | str] = None,
    overrides_path: Optional[Path | str] = None,
) -> AgentSpec:
    """
    Load spec/flight_booking.json and apply prompt_overrides.yaml in memory.
    """
    spec_file = Path(spec_path) if spec_path else DEFAULT_SPEC_PATH
    overrides_file = Path(overrides_path) if overrides_path else DEFAULT_OVERRIDES_PATH

    with open(spec_file, "r", encoding="utf-8") as f:
        raw_spec = json.load(f)

    spec_data = copy.deepcopy(raw_spec)
    overrides: Dict[str, Any] = {}
    if overrides_file.exists():
        with open(overrides_file, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}

    nodes_by_name: Dict[str, Dict[str, Any]] = {
        node["name"]: node for node in spec_data.get("nodes", [])
    }

    start_node = nodes_by_name["Start"]["parameters"]
    agent_node = nodes_by_name["eandAI"]["parameters"]
    stt_node = nodes_by_name["Breez Ears"]["parameters"]
    llm_node = nodes_by_name["OpenAI Brain"]["parameters"]
    tts_node = nodes_by_name["ElevenLabs Voice"]["parameters"]
    bg_node = nodes_by_name["BGNoise_Office"]["parameters"]
    search_node = nodes_by_name["SearchFlights"]["parameters"]
    kb_node = nodes_by_name["KBFlight"]["parameters"]
    resp_node = nodes_by_name["RESPFlight"]["parameters"]

    applied_overrides: List[str] = []

    # 1. Apply persona standardization overrides
    persona_cfg = overrides.get("persona_standardization", {})
    replacements_cfg = persona_cfg.get("replacements", {})

    for rule in replacements_cfg.get("Start.parameters.masterInstruction", []):
        src, dst = rule.get("from", ""), rule.get("to", "")
        if src and src in start_node["masterInstruction"]:
            start_node["masterInstruction"] = start_node["masterInstruction"].replace(src, dst)
            applied_overrides.append(
                f"Start.masterInstruction: replaced '{src}' -> '{dst}'"
            )

    for rule in replacements_cfg.get("Start.parameters.inputs[caller_name].description", []):
        src, dst = rule.get("from", ""), rule.get("to", "")
        for inp in start_node.get("inputs", []):
            if inp.get("variable_name") == "caller_name" and src in inp.get("description", ""):
                inp["description"] = inp["description"].replace(src, dst)
                applied_overrides.append(
                    f"Start.inputs[caller_name].description: replaced '{src}' -> '{dst}'"
                )

    # 2. Apply input mapping overrides
    input_mappings = overrides.get("input_mappings", {})
    if input_mappings:
        applied_overrides.append(
            "Input mappings: mapped {{customer_name}} <-> caller_name and added optional enquiry_route, phone_number"
        )

    # 3. Apply SearchFlights tool overrides
    tool_overrides = overrides.get("tool_overrides", {}).get("SearchFlights", {})
    query_override_desc = (
        tool_overrides.get("inputs", {}).get("query", {}).get("description")
    )
    query_desc = ""
    for inp in search_node.get("inputs", []):
        if inp.get("variable_name") == "query":
            if query_override_desc:
                inp["description"] = query_override_desc
                applied_overrides.append(
                    "SearchFlights.inputs[query].description: replaced real-estate examples with flight examples"
                )
            query_desc = inp.get("description", "")

    # 4. Apply STT language hint overrides
    stt_override = overrides.get("stt_overrides", {}).get("Breez Ears", {})
    language_hints: List[str] = list(stt_node.get("language_hints", []))
    for lang in stt_override.get("add_language_hints", []):
        if lang not in language_hints:
            language_hints.append(lang)
            applied_overrides.append(f"Breez Ears.language_hints: added '{lang}'")

    # Build STTConfig
    stt_context = stt_node.get("context", "")
    stt_config = STTConfig(
        provider_model=stt_node.get("model", "echo-universal"),
        context=stt_context,
        keyterms=extract_keyterms_from_context(stt_context),
        language_hints=language_hints,
        max_endpoint_delay_ms=int(stt_node.get("max_endpoint_delay_ms", 1720)),
        endpoint_sensitivity=float(stt_node.get("endpoint_sensitivity", 0.15)),
        enable_diarization=bool(stt_node.get("enable_diarization", False)),
    )

    # Build LLMConfig
    raw_llm_model = llm_node.get("model", "gpt-5.4-mini")
    normalized_llm_model = os.getenv("OPENAI_MODEL") or _normalize_openai_model(raw_llm_model)
    llm_config = LLMConfig(
        raw_model=raw_llm_model,
        model=normalized_llm_model,
        reasoning_effort=str(llm_node.get("reasoning_effort_5_4", "none")),
        max_completion_tokens=int(llm_node.get("max_completion_tokens_gpt5", 500)),
    )

    # Build TTSConfig
    default_voice_id = tts_node.get("custom_voice_id", "ack0QsRaQyDLnVyMQTSd")
    voice_id = os.getenv("ELEVEN_VOICE_ID", "").strip() or default_voice_id
    tts_config = TTSConfig(
        model=tts_node.get("model", "eleven_flash_v2_5"),
        voice_id=voice_id,
        stability=float(tts_node.get("stability", 0.4)),
        similarity_boost=float(tts_node.get("similarity_boost", 0.8)),
        style=float(tts_node.get("style", 0.35)),
        use_speaker_boost=bool(tts_node.get("use_speaker_boost", True)),
        speed=float(tts_node.get("speed", 1.0)),
    )

    # Build TurnTakingConfig (eandAI overrides Start when override flags are true)
    min_interruption_words = int(start_node.get("minInterruptionWords", 3))
    if agent_node.get("overrideMinInterruptionWords"):
        min_interruption_words = int(
            agent_node.get("minInterruptionWords", min_interruption_words)
        )

    max_endpointing_delay = float(start_node.get("maxEndpointingDelay", 2.0))
    if agent_node.get("overrideMaxEndpointingDelay"):
        max_endpointing_delay = float(
            agent_node.get("maxEndpointingDelay", max_endpointing_delay)
        )

    turn_taking = TurnTakingConfig(
        min_interruption_words=min_interruption_words,
        max_endpointing_delay=max_endpointing_delay,
        stt_max_endpoint_delay_ms=stt_config.max_endpoint_delay_ms,
        speak_first_mode=str(agent_node.get("speakFirstMode", "off")),
    )

    # Build CallLimitsConfig
    limits = CallLimitsConfig(
        max_call_duration_enabled=bool(start_node.get("maxCallDurationEnabled", True)),
        max_call_duration_minutes=float(start_node.get("maxCallDurationMinutes", 10)),
        silence_timeout_seconds=float(start_node.get("silenceTimeoutSeconds", 30)),
    )

    # Build BackgroundNoiseConfig
    bg_noise = BackgroundNoiseConfig(
        enabled=bg_node.get("operation", "on") == "on",
        track=str(bg_node.get("track", "office_chatter")),
        volume_percent=int(bg_node.get("volume", 20)),
    )

    # Build FillersConfig
    fillers = FillersConfig(
        enabled=bool(start_node.get("fillers_enabled", True)),
        fillers=list(start_node.get("naturalness_fillers", [])),
        language=str(start_node.get("naturalness_language", "en")),
    )

    # Build SearchToolConfig
    search_tool = SearchToolConfig(
        name="search_flights",
        flow_name=str(search_node.get("flow_name", "search_Flights")),
        instructions=str(search_node.get("instructions", "")),
        query_description=query_desc,
    )

    # Build KnowledgeBaseConfig
    kb_options = kb_node.get("options", {})
    kb_config = KnowledgeBaseConfig(
        knowledge_base_id=str(kb_node.get("knowledgeBaseId", "")),
        top_k=int(kb_node.get("topK", 5)),
        model=str(kb_options.get("model", "openai:gpt-5.4-nano")),
        use_reranker=bool(kb_options.get("useReranker", True)),
        decompose_query=bool(kb_options.get("decomposeQuery", True)),
        rerank_candidates=int(kb_options.get("rerankCandidates", 30)),
        output_mode=str(kb_node.get("outputMode", "chunks")),
        score_threshold=float(kb_node.get("scoreThreshold", 0.0)),
        raw_inventory=str(resp_node.get("rawResponse", "")),
    )

    # Build PostCallConfig
    post_call = PostCallConfig(
        model=_normalize_openai_model(str(start_node.get("post_call_model", "gpt-5.4-mini"))),
        reasoning_effort=str(start_node.get("post_call_reasoning_effort", "high")),
    )

    return AgentSpec(
        workflow_id=str(spec_data.get("workflowId") or spec_data.get("id", "")),
        workflow_name=str(spec_data.get("name", "flight booking")),
        master_instruction=str(start_node.get("masterInstruction", "")),
        agent_instructions=str(agent_node.get("instructions", "")),
        stt=stt_config,
        llm=llm_config,
        tts=tts_config,
        turn_taking=turn_taking,
        limits=limits,
        bg_noise=bg_noise,
        fillers=fillers,
        search_tool=search_tool,
        kb=kb_config,
        post_call=post_call,
        applied_overrides=applied_overrides,
    )


if __name__ == "__main__":
    spec = load_spec()
    print(f"Loaded workflow: {spec.workflow_name} ({spec.workflow_id})")
    print(f"LLM: {spec.llm.model} (raw: {spec.llm.raw_model}), max_tokens={spec.llm.max_completion_tokens}, reasoning={spec.llm.reasoning_effort}")
    print(f"TTS: {spec.tts.model}, voice={spec.tts.voice_id}, stability={spec.tts.stability}, similarity={spec.tts.similarity_boost}, style={spec.tts.style}, speaker_boost={spec.tts.use_speaker_boost}, speed={spec.tts.speed}")
    print(f"STT languages: {spec.stt.language_hints}, keyterms count: {len(spec.stt.keyterms)}, max_endpoint_delay_ms={spec.stt.max_endpoint_delay_ms}")
    print(f"Turn taking: min_words={spec.turn_taking.min_interruption_words}, max_delay={spec.turn_taking.max_endpointing_delay}s")
    print(f"Overrides applied ({len(spec.applied_overrides)}):")
    for ov in spec.applied_overrides:
        print(f"  - {ov}")

