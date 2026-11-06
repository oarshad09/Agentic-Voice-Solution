"""
STT Adapter for LiveKit Agents 1.x.

Maps the `Breez Ears` node from `spec/flight_booking.json` to either:
1. Deepgram Nova-3 (`nova-3` with `language="multi"`, keyterms from `Breez Ears.context`,
   and endpointing delay `1720ms` from `max_endpoint_delay_ms`) — default.
2. Soniox (`soniox.STT` with `language_hints=["en", "ar", "hi", "ur", "es"]`, context/keyterms,
   and endpoint delay `1720ms`), selectable via `STT_PROVIDER=soniox`.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Optional

from spec_loader import STTConfig

logger = logging.getLogger("ezee-flights.stt")


def build_stt(stt_cfg: STTConfig, provider: Optional[str] = None) -> Any:
    """
    Create a LiveKit STT instance from `STTConfig` using the configured provider
    ('deepgram' by default, or 'soniox').
    """
    chosen = (provider or os.getenv("STT_PROVIDER", "deepgram")).strip().lower()

    if chosen == "soniox":
        return _build_soniox_stt(stt_cfg)
    return _build_deepgram_stt(stt_cfg)


def _build_deepgram_stt(stt_cfg: STTConfig) -> Any:
    from livekit.plugins import deepgram

    sig = inspect.signature(deepgram.STT.__init__)
    params = sig.parameters
    kwargs: dict[str, Any] = {
        "model": os.getenv("DEEPGRAM_MODEL", "nova-3"),
        "language": os.getenv("DEEPGRAM_LANGUAGE", "multi"),
    }

    # Pass keyterms (Nova-3) or keywords (Nova-2 fallback) based on plugin signature
    if "keyterms" in params and stt_cfg.keyterms:
        kwargs["keyterms"] = stt_cfg.keyterms
    elif "keyterm" in params and stt_cfg.keyterms:
        kwargs["keyterm"] = stt_cfg.keyterms
    elif "keywords" in params and stt_cfg.keyterms:
        kwargs["keywords"] = [(term, 1.5) for term in stt_cfg.keyterms]

    # Map max_endpoint_delay_ms (1720 ms)
    if "endpointing_ms" in params:
        kwargs["endpointing_ms"] = stt_cfg.max_endpoint_delay_ms
    elif " interim_results" in params:
        kwargs["interim_results"] = True

    if "smart_format" in params:
        kwargs["smart_format"] = True

    logger.info(
        "Initialized Deepgram STT (model=%s, language=%s, keyterms=%d, endpointing_ms=%d)",
        kwargs.get("model"),
        kwargs.get("language"),
        len(stt_cfg.keyterms),
        stt_cfg.max_endpoint_delay_ms,
    )
    return deepgram.STT(**kwargs)


def _build_soniox_stt(stt_cfg: STTConfig) -> Any:
    try:
        from livekit.plugins import soniox
    except ImportError as exc:
        raise RuntimeError(
            "STT_PROVIDER=soniox requires `livekit-plugins-soniox` to be installed."
        ) from exc

    sig = inspect.signature(soniox.STT.__init__)
    params = sig.parameters
    kwargs: dict[str, Any] = {}

    # Soniox context / keyterms + language hints (en, ar, hi, ur, es)
    if "params" in params and hasattr(soniox, "STTOptions"):
        opt_sig = inspect.signature(soniox.STTOptions.__init__).parameters
        opt_kwargs: dict[str, Any] = {}
        if "language_hints" in opt_sig:
            opt_kwargs["language_hints"] = stt_cfg.language_hints
        if "context" in opt_sig:
            opt_kwargs["context"] = stt_cfg.context
        if "max_non_final_tokens_duration_ms" in opt_sig:
            opt_kwargs["max_non_final_tokens_duration_ms"] = stt_cfg.max_endpoint_delay_ms
        kwargs["params"] = soniox.STTOptions(**opt_kwargs)
    else:
        if "language_hints" in params:
            kwargs["language_hints"] = stt_cfg.language_hints
        if "context" in params:
            kwargs["context"] = stt_cfg.context
        if "keyterms" in params:
            kwargs["keyterms"] = stt_cfg.keyterms
        if "max_endpoint_delay_ms" in params:
            kwargs["max_endpoint_delay_ms"] = stt_cfg.max_endpoint_delay_ms

    logger.info(
        "Initialized Soniox STT (language_hints=%s, keyterms=%d, max_endpoint_delay_ms=%d)",
        stt_cfg.language_hints,
        len(stt_cfg.keyterms),
        stt_cfg.max_endpoint_delay_ms,
    )
    return soniox.STT(**kwargs)

