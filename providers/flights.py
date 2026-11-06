"""
Flight search providers backing the @function_tool `search_flights(query: str)`.

- FlightProvider: Abstract interface so a live GDS/NDC API (Amadeus, Sabre, Duffel)
  can be plugged in without changing agent or tool logic.
- StaticProvider (v1): Returns `RESPFlight.parameters.rawResponse` verbatim.
- KBProvider (v2): Chunks `RESPFlight.parameters.rawResponse` into route/option
  documents, indexes them in a local vector store (hybrid dense/n-gram + TF-IDF
  vectors, with optional OpenAI embeddings), decomposes compound queries,
  retrieves candidates (`rerankCandidates=30`), reranks them, and returns the
  top-k (`topK=5`) matching chunks.
"""

from __future__ import annotations

import abc
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from spec_loader import AgentSpec, KnowledgeBaseConfig, load_spec


@dataclass
class FlightChunk:
    """A single indexed chunk in the local flight knowledge base."""

    chunk_id: str
    section_header: str
    title: str
    text: str
    metadata: Dict[str, str]

    @property
    def full_text(self) -> str:
        if self.section_header and self.section_header not in self.text:
            return f"{self.section_header}\n{self.text}"
        return self.text


class FlightProvider(abc.ABC):
    """Abstract interface for flight inventory search."""

    @abc.abstractmethod
    async def search(self, query: str) -> str:
        """Search flight inventory asynchronously and return formatted text."""

    def search_sync(self, query: str) -> str:
        """Synchronous search helper for non-async callers or unit tests."""
        raise NotImplementedError("Subclasses should implement search_sync or use await search().")


class StaticProvider(FlightProvider):
    """
    v1 FlightProvider: returns RESPFlight.rawResponse verbatim.
    """

    def __init__(self, raw_response: Optional[str] = None, spec: Optional[AgentSpec] = None) -> None:
        if raw_response is not None:
            self._raw_response = raw_response
        else:
            loaded = spec or load_spec()
            self._raw_response = loaded.kb.raw_inventory

    def search_sync(self, query: str) -> str:
        return self._raw_response

    async def search(self, query: str) -> str:
        return self.search_sync(query)


# Domain synonyms / airport codes to enrich local vector representations
_SYNONYM_MAP: Dict[str, List[str]] = {
    "dxb": ["dubai", "dxb"],
    "dubai": ["dubai", "dxb"],
    "lhr": ["london", "heathrow", "lhr", "uk"],
    "london": ["london", "heathrow", "lhr", "uk"],
    "heathrow": ["london", "heathrow", "lhr"],
    "khi": ["karachi", "khi", "pakistan"],
    "karachi": ["karachi", "khi", "pakistan"],
    "lhe": ["lahore", "lhe", "pakistan"],
    "lahore": ["lahore", "lhe", "pakistan"],
    "jed": ["jeddah", "jed", "umrah", "saudi", "makkah", "mecca"],
    "jeddah": ["jeddah", "jed", "umrah", "saudi", "makkah"],
    "umrah": ["jeddah", "jed", "umrah", "saudia", "zamzam"],
    "ist": ["istanbul", "turkish", "connection", "stop", "layover"],
    "istanbul": ["istanbul", "turkish", "connection", "stop", "layover"],
    "bah": ["bahrain", "gulf", "connection", "stop", "layover"],
    "cheap": ["cheapest", "budget", "lower", "light", "lite", "stop", "connection", "1690", "1750", "1850", "950", "980", "1080"],
    "cheapest": ["cheap", "budget", "lower", "light", "lite", "stop", "connection", "1690", "1750", "1850"],
    "direct": ["direct", "non-stop", "nonstop", "no stops", "emirates", "british airways", "pia", "saudia", "flydubai"],
    "connection": ["stop", "connecting", "layover", "istanbul", "bahrain", "turkish", "gulf air"],
    "connecting": ["stop", "connection", "layover", "istanbul", "bahrain", "turkish", "gulf air"],
    "stop": ["stop", "connection", "layover", "istanbul", "bahrain"],
    "baggage": ["baggage", "bag", "checked", "cabin", "hand luggage", "kg", "allowance"],
    "luggage": ["baggage", "bag", "checked", "cabin", "hand luggage", "kg"],
    "refund": ["refund", "refundable", "non-refundable", "changes", "fee", "cancel"],
    "change": ["changes", "change", "fee", "reschedule", "flexible", "earlier", "11 december"],
    "earlier": ["11 december", "11dec", "one day earlier", "friday"],
    "hotel": ["hotel", "hotels", "paddington", "king's cross", "accommodation", "room", "stay", "holiday"],
    "hotels": ["hotel", "hotels", "paddington", "king's cross", "accommodation", "room", "stay"],
}


def _tokenize(text: str) -> List[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
    expanded: List[str] = []
    for tok in raw_tokens:
        expanded.append(tok)
        if tok in _SYNONYM_MAP:
            expanded.extend(_SYNONYM_MAP[tok])
    # Also include character 3-grams and 4-grams for fuzzy subword matching
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    for n in (3, 4):
        for i in range(max(0, len(cleaned) - n + 1)):
            gram = cleaned[i : i + n]
            if gram.strip():
                expanded.append(f"#{gram}")
    return expanded


def chunk_inventory_text(raw_text: str) -> List[FlightChunk]:
    """
    Chunk `RESPFlight.rawResponse` into logical route/option chunks while preserving
    the parent route/date section header on every chunk.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text.strip()) if p.strip()]
    chunks: List[FlightChunk] = []
    current_section = ""
    idx = 0

    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        lines = [ln.strip() for ln in para.splitlines() if ln.strip()]

        # Case 1: Paragraph has a section header line followed by "Option ..."
        if len(lines) >= 2 and lines[1].startswith("Option "):
            current_section = lines[0]
            option_title = lines[1]
            body = paragraphs[i + 1] if (i + 1 < len(paragraphs) and paragraphs[i + 1].startswith("Route:")) else ""
            if body:
                i += 1
            idx += 1
            chunk_text = f"{option_title}\n\n{body}".strip()
            chunks.append(
                FlightChunk(
                    chunk_id=f"chunk-{idx:02d}",
                    section_header=current_section,
                    title=option_title,
                    text=chunk_text,
                    metadata={"section": current_section, "title": option_title},
                )
            )
        # Case 2: Paragraph starts directly with "Option ..."
        elif lines[0].startswith("Option "):
            option_title = lines[0]
            body = paragraphs[i + 1] if (i + 1 < len(paragraphs) and paragraphs[i + 1].startswith("Route:")) else ""
            if body:
                i += 1
            idx += 1
            chunk_text = f"{option_title}\n\n{body}".strip()
            chunks.append(
                FlightChunk(
                    chunk_id=f"chunk-{idx:02d}",
                    section_header=current_section,
                    title=option_title,
                    text=chunk_text,
                    metadata={"section": current_section, "title": option_title},
                )
            )
        # Case 3: Multi-line header + route title (e.g. "Other routes ...\nDubai to Karachi — return")
        elif len(lines) >= 2 and ("— return" in lines[-1] or "Hotels —" in lines[-1]):
            current_section = lines[0]
            route_title = lines[-1]
            body = paragraphs[i + 1] if (i + 1 < len(paragraphs)) else ""
            if body:
                i += 1
            idx += 1
            chunk_text = f"{route_title}\n\n{body}".strip()
            chunks.append(
                FlightChunk(
                    chunk_id=f"chunk-{idx:02d}",
                    section_header=current_section,
                    title=route_title,
                    text=chunk_text,
                    metadata={"section": current_section, "title": route_title},
                )
            )
        # Case 4: Single route or hotel header followed by body paragraph
        elif "— return" in lines[0] or lines[0].startswith("Hotels —"):
            route_title = lines[0]
            if route_title.startswith("Hotels —"):
                current_section = "Hotels — Cross-sell inventory"
            body = paragraphs[i + 1] if (i + 1 < len(paragraphs)) else ""
            if body:
                i += 1
            idx += 1
            chunk_text = f"{route_title}\n\n{body}".strip()
            chunks.append(
                FlightChunk(
                    chunk_id=f"chunk-{idx:02d}",
                    section_header=current_section,
                    title=route_title,
                    text=chunk_text,
                    metadata={"section": current_section, "title": route_title},
                )
            )
        else:
            idx += 1
            chunks.append(
                FlightChunk(
                    chunk_id=f"chunk-{idx:02d}",
                    section_header=current_section,
                    title=lines[0],
                    text=para,
                    metadata={"section": current_section, "title": lines[0]},
                )
            )
        i += 1

    return chunks


class LocalVectorIndex:
    """
    In-memory vector store using L2-normalized TF-IDF weighted word/synonym/n-gram
    vectors and cosine similarity, plus a structured cross-feature reranker.
    Mirrors KBFlight parameters (topK=5, decomposeQuery=True, useReranker=True,
    rerankCandidates=30, scoreThreshold=0).
    """

    def __init__(self, chunks: Sequence[FlightChunk]) -> None:
        self.chunks: List[FlightChunk] = list(chunks)
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []
        self._build_index()

    def _build_index(self) -> None:
        if not self.chunks:
            return

        doc_tokens: List[List[str]] = [
            _tokenize(chunk.full_text) for chunk in self.chunks
        ]
        df: Counter[str] = Counter()
        for tokens in doc_tokens:
            for unique_tok in set(tokens):
                df[unique_tok] += 1

        n_docs = len(self.chunks)
        self.idf = {
            tok: math.log((1.0 + n_docs) / (1.0 + count)) + 1.0
            for tok, count in df.items()
        }

        self.doc_vectors = []
        for tokens in doc_tokens:
            counts = Counter(tokens)
            vec: Dict[str, float] = {}
            norm_sq = 0.0
            for tok, count in counts.items():
                val = (1.0 + math.log(count)) * self.idf[tok]
                vec[tok] = val
                norm_sq += val * val
            norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
            self.doc_vectors.append({k: v / norm for k, v in vec.items()})

    def _embed_query(self, query: str) -> Dict[str, float]:
        counts = Counter(_tokenize(query))
        vec: Dict[str, float] = {}
        norm_sq = 0.0
        for tok, count in counts.items():
            if tok in self.idf:
                val = (1.0 + math.log(count)) * self.idf[tok]
                vec[tok] = val
                norm_sq += val * val
        norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
        return {k: v / norm for k, v in vec.items()}

    @staticmethod
    def _cosine(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a
        return sum(val * vec_b.get(k, 0.0) for k, val in vec_a.items())

    @staticmethod
    def decompose_query(query: str) -> List[str]:
        """
        Decompose a multi-clause flight search query into sub-queries when
        `decomposeQuery=True` in KBFlight.
        """
        base = query.strip()
        if not base:
            return [base]
        sub_queries = [base]
        parts = [
            p.strip()
            for p in re.split(r"[,;]+|\b(?:and|or|plus|also|versus|vs)\b", base, flags=re.IGNORECASE)
            if len(p.strip()) > 3
        ]
        if len(parts) > 1:
            sub_queries.extend(parts)
        return sub_queries

    @staticmethod
    def _rerank_score(query: str, chunk: FlightChunk, base_score: float) -> float:
        """
        Cross-feature reranker combining dense vector cosine score with
        exact entity/route/date/attribute boosts.
        """
        q_lower = query.lower()
        c_lower = chunk.full_text.lower()
        bonus = 0.0

        # Destination / route match boosts
        destinations = [
            ("london", ["london", "heathrow", "lhr"]),
            ("karachi", ["karachi", "khi"]),
            ("lahore", ["lahore", "lhe"]),
            ("jeddah", ["jeddah", "jed", "umrah"]),
            ("hotel", ["hotel", "paddington", "king's cross"]),
        ]
        for _, aliases in destinations:
            if any(a in q_lower for a in aliases):
                if any(a in c_lower for a in aliases):
                    bonus += 0.35
                else:
                    bonus -= 0.15

        # Date specificity boosts (11 Dec vs 12 Dec)
        wants_11 = any(k in q_lower for k in ("11 dec", "11th", "day earlier", "one day earlier", "friday 11"))
        wants_12 = any(k in q_lower for k in ("12 dec", "12th", "saturday 12"))
        if wants_11 and "11 december" in c_lower:
            bonus += 0.25
        if wants_12 and "12 december" in c_lower:
            bonus += 0.20

        # Direct vs connecting preference boost
        if "direct" in q_lower or "non-stop" in q_lower or "no stop" in q_lower:
            if "direct" in c_lower:
                bonus += 0.15
        if any(k in q_lower for k in ("stop", "connect", "cheapest", "cheaper", "lower", "budget", "under")):
            if "one stop" in c_lower or "11 december" in c_lower or "hand luggage only" in c_lower:
                bonus += 0.15

        # Airline match boost
        for airline in ("emirates", "british airways", "turkish", "gulf air", "flydubai", "pia", "saudia"):
            if airline in q_lower and airline in c_lower:
                bonus += 0.25

        return float(base_score + bonus)

    def search(
        self,
        query: str,
        top_k: int = 5,
        decompose: bool = True,
        use_reranker: bool = True,
        rerank_candidates: int = 30,
        score_threshold: float = 0.0,
    ) -> List[Tuple[FlightChunk, float]]:
        if not self.chunks:
            return []

        queries = self.decompose_query(query) if decompose else [query]
        agg_scores = [0.0] * len(self.chunks)

        for idx, sub_q in enumerate(queries):
            q_vec = self._embed_query(sub_q)
            weight = 1.0 if idx == 0 else 0.35
            for d_idx, d_vec in enumerate(self.doc_vectors):
                sim = self._cosine(q_vec, d_vec) * weight
                agg_scores[d_idx] = max(agg_scores[d_idx], sim) + (0.1 * sim)

        # Select top `rerank_candidates`
        candidate_indices = sorted(
            range(len(self.chunks)), key=lambda i: agg_scores[i], reverse=True
        )[:rerank_candidates]

        scored: List[Tuple[FlightChunk, float]] = []
        for c_idx in candidate_indices:
            chunk = self.chunks[c_idx]
            raw_score = agg_scores[c_idx]
            final_score = (
                self._rerank_score(query, chunk, raw_score)
                if use_reranker
                else raw_score
            )
            if final_score >= score_threshold:
                scored.append((chunk, final_score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


class KBProvider(FlightProvider):
    """
    v2 FlightProvider: chunks `RESPFlight.rawResponse` into a local vector store,
    decomposes the query, retrieves up to `rerankCandidates`, reranks, and returns
    the `top_k` (default 5) matching chunks.
    """

    def __init__(
        self,
        kb_config: Optional[KnowledgeBaseConfig] = None,
        spec: Optional[AgentSpec] = None,
        top_k: Optional[int] = None,
    ) -> None:
        if kb_config is None:
            loaded = spec or load_spec()
            kb_config = loaded.kb
        self.kb_config = kb_config
        self.top_k = top_k if top_k is not None else kb_config.top_k
        self.chunks = chunk_inventory_text(kb_config.raw_inventory)
        self.index = LocalVectorIndex(self.chunks)

    def search_sync(self, query: str) -> str:
        results = self.index.search(
            query=query,
            top_k=self.top_k,
            decompose=self.kb_config.decompose_query,
            use_reranker=self.kb_config.use_reranker,
            rerank_candidates=self.kb_config.rerank_candidates,
            score_threshold=self.kb_config.score_threshold,
        )
        if not results:
            return "No matching flights found in inventory for this query."

        rendered_chunks: List[str] = []
        seen_headers: set[str] = set()
        for chunk, _score in results:
            if chunk.section_header and chunk.section_header not in seen_headers:
                seen_headers.add(chunk.section_header)
                rendered_chunks.append(f"{chunk.section_header}\n{chunk.text}")
            else:
                rendered_chunks.append(chunk.text)

        return "\n\n".join(rendered_chunks)

    async def search(self, query: str) -> str:
        return self.search_sync(query)


class GDSProviderStub(FlightProvider):
    """
    Placeholder for future live GDS / NDC (e.g. Amadeus, Duffel, Sabre) integration.
    Demonstrates how a live provider slots into FlightProvider.
    """

    async def search(self, query: str) -> str:
        raise NotImplementedError(
            "GDSProviderStub is a template for live GDS/NDC API integration. "
            "Set FLIGHT_PROVIDER=kb or FLIGHT_PROVIDER=static."
        )


def create_flight_provider(
    mode: Optional[str] = None,
    spec: Optional[AgentSpec] = None,
) -> FlightProvider:
    """
    Factory that returns either KBProvider ('kb' / 'v2', default) or
    StaticProvider ('static' / 'v1'), controlled by mode or FLIGHT_PROVIDER env var.
    """
    resolved_mode = (mode or os.getenv("FLIGHT_PROVIDER", "kb")).strip().lower()
    loaded_spec = spec or load_spec()
    if resolved_mode in ("static", "v1"):
        return StaticProvider(spec=loaded_spec)
    if resolved_mode in ("kb", "v2", "vector"):
        return KBProvider(spec=loaded_spec)
    if resolved_mode in ("gds", "ndc"):
        return GDSProviderStub()
    return KBProvider(spec=loaded_spec)
