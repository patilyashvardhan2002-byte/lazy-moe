"""
LazyMoE Query Analyzer
Predicts which MoE experts will activate for a given query,
so we can prefetch them before inference begins.

Two modes:
  - 'keyword'  : fast heuristic, no dependencies
  - 'embedding': uses sentence-transformers for accurate prediction
                 (requires: pip install sentence-transformers)
"""

import re
import time
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("lazy-moe.analyzer")

# Expert activation profile per domain
# Based on profiling Mixtral 8x22B on domain-specific datasets
# Replace with your own profiling data for accuracy
EXPERT_PROFILES = {
    "code": {
        "experts": [0, 3, 5, 7],
        "keywords": ["function","def","class","code","debug","python","javascript",
                     "typescript","rust","c++","algorithm","implement","bug","error",
                     "compile","syntax","loop","array","api","database","sql","git",
                     "docker","import","return","variable","async","await","null"],
    },
    "math": {
        "experts": [1, 2, 5, 6],
        "keywords": ["calculate","solve","equation","integral","derivative","matrix",
                     "probability","statistics","sum","formula","proof","theorem",
                     "algebra","calculus","geometry","graph","function","series",
                     "limit","infinity","modulo","prime","factorial","vector"],
    },
    "language": {
        "experts": [0, 2, 4, 6],
        "keywords": ["translate","grammar","write","essay","summarize","explain",
                     "describe","language","word","sentence","paragraph","meaning",
                     "definition","synonym","antonym","metaphor","poetry","literature",
                     "french","spanish","german","chinese","japanese","arabic"],
    },
    "reasoning": {
        "experts": [1, 3, 4, 7],
        "keywords": ["why","how","analyze","compare","evaluate","logic","reason",
                     "think","argument","cause","effect","conclusion","because",
                     "therefore","however","evidence","hypothesis","infer","deduce",
                     "consequence","implication","tradeoff","pros","cons"],
    },
    "creative": {
        "experts": [0, 2, 4, 5],
        "keywords": ["story","poem","creative","imagine","design","invent","brainstorm",
                     "idea","fiction","write a","generate","create","character","plot",
                     "setting","dialogue","narrative","metaphor","fantasy","sci-fi"],
    },
    "factual": {
        "experts": [2, 3, 6, 7],
        "keywords": ["what is","who is","when","where","history","fact","define",
                     "explain what","tell me about","information","wiki","overview",
                     "summary of","background","context","founded","invented"],
    },
}


@dataclass
class QueryAnalysis:
    domain: str
    confidence: float
    active_experts: list[int]   # predicted experts (top-2 for Mixtral)
    fallback_experts: list[int] # secondary experts to prefetch
    tokens_estimate: int
    analysis_ms: float


class QueryAnalyzer:
    """
    Predicts expert activation from a query before inference.
    Supports keyword heuristics (fast) and embedding similarity (accurate).
    """

    def __init__(self, mode: str = "keyword", model_name: Optional[str] = None):
        self.mode = mode
        self._embedder = None
        self._profile_embeddings = None

        if mode == "embedding":
            self._init_embedder(model_name or "all-MiniLM-L6-v2")

    def analyze(self, query: str) -> QueryAnalysis:
        t0 = time.perf_counter()

        if self.mode == "embedding" and self._embedder is not None:
            domain, confidence = self._embedding_classify(query)
        else:
            domain, confidence = self._keyword_classify(query)

        profile = EXPERT_PROFILES[domain]
        experts = profile["experts"]

        # Active = top 2 (matches Mixtral routing: top-2 experts per token)
        active = experts[:2]
        # Fallback = remaining from profile (prefetch candidates)
        fallback = experts[2:]

        elapsed = (time.perf_counter() - t0) * 1000

        analysis = QueryAnalysis(
            domain=domain,
            confidence=confidence,
            active_experts=active,
            fallback_experts=fallback,
            tokens_estimate=self._estimate_tokens(query),
            analysis_ms=round(elapsed, 2),
        )

        logger.info(
            f"Query analyzed: domain={domain} conf={confidence:.2f} "
            f"experts={active} time={elapsed:.1f}ms"
        )
        return analysis

    # ── Keyword classifier ────────────────────────────────────────────────────

    def _keyword_classify(self, query: str) -> tuple[str, float]:
        q = query.lower()
        scores: dict[str, int] = {}

        for domain, profile in EXPERT_PROFILES.items():
            score = sum(1 for kw in profile["keywords"] if kw in q)
            scores[domain] = score

        total = sum(scores.values())
        if total == 0:
            return "reasoning", 0.4  # default

        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_domain, top_score = sorted_domains[0]

        # Confidence = normalized score, clamped
        confidence = min(0.5 + (top_score / max(total, 1)) * 0.5, 0.97)
        return top_domain, confidence

    # ── Embedding classifier ──────────────────────────────────────────────────

    def _init_embedder(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            logger.info(f"Loading embedding model: {model_name}")
            self._embedder = SentenceTransformer(model_name)

            # Build reference embeddings for each domain
            domain_descriptions = {
                "code":      "Write code, debug, implement algorithm, programming function class",
                "math":      "Calculate solve equation integral derivative probability statistics",
                "language":  "Translate grammar write essay summarize explain language meaning",
                "reasoning": "Analyze compare evaluate logic argument cause effect conclusion",
                "creative":  "Story poem creative imagine design fiction narrative character",
                "factual":   "What is who when where history fact define explain information",
            }
            self._profile_embeddings = {
                domain: self._embedder.encode(desc)
                for domain, desc in domain_descriptions.items()
            }
            logger.info("Embedding classifier ready")
        except ImportError:
            logger.warning("sentence-transformers not installed — falling back to keyword mode")
            self.mode = "keyword"

    def _embedding_classify(self, query: str) -> tuple[str, float]:
        import numpy as np

        q_emb = self._embedder.encode(query)
        scores = {}

        for domain, ref_emb in self._profile_embeddings.items():
            # Cosine similarity
            sim = float(np.dot(q_emb, ref_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(ref_emb)))
            scores[domain] = sim

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_domain, top_score = sorted_scores[0]

        # Normalize confidence
        min_s, max_s = sorted_scores[-1][1], top_score
        confidence = (top_score - min_s) / (max_s - min_s + 1e-8)
        confidence = 0.5 + confidence * 0.47  # remap to [0.5, 0.97]

        return top_domain, round(confidence, 3)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _estimate_tokens(self, query: str) -> int:
        """Rough token estimate: ~4 chars per token for English."""
        words = len(query.split())
        chars = len(query)
        return max(int(chars / 4), words)
