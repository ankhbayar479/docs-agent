"""Hybrid, two-stage retrieval.

1. Score every chunk with a weighted mix of vector similarity and BM25 keyword score.
2. Stage 1: rank whole pages, so one strong page beats many weak, scattered chunks.
3. Stage 2: take candidate chunks only from the top pages, always including a few
   from the best page, then (optionally) rerank them with Cohere Rerank.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .config import Settings
from .index import Chunk, Embedder
from .text import tokenize


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]: ...


class CohereReranker:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        resp = self.client.rerank(model=self.model, query=query, documents=documents, top_n=top_n)
        return [(r.index, r.relevance_score) for r in resp.results]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # final score used for ordering (rerank score if reranked)
    hybrid_score: float


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        embedder: Embedder,
        settings: Settings,
        reranker: Reranker | None = None,
    ):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings
        self.bm25 = BM25Okapi([tokenize(c.embed_text()) for c in chunks])
        self.by_section: dict[str, list[Chunk]] = defaultdict(list)
        self.by_page: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            self.by_section[c.section_id].append(c)
            self.by_page[c.page_id].append(c)

    # ---- scoring -------------------------------------------------------------
    def hybrid_scores(self, query: str, lang: str | None) -> np.ndarray:
        q = self.embedder.embed([query], input_type="search_query")[0]
        semantic = self.vectors @ q
        keyword = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
        mask = np.ones(len(self.chunks), dtype=bool)
        if lang in ("en", "ja"):
            mask = np.array([c.lang == lang for c in self.chunks])
        scores = np.full(len(self.chunks), -1.0, dtype=np.float32)
        if mask.any():
            s = self.settings
            scores[mask] = s.semantic_weight * _minmax(semantic[mask]) + s.keyword_weight * _minmax(keyword[mask])
        return scores

    @staticmethod
    def rank_pages(chunks: list[Chunk], scores: np.ndarray) -> list[tuple[str, float]]:
        """Page score = best chunk + a quarter of the second best: rewards pages with more than one match."""
        per_page: dict[str, list[float]] = defaultdict(list)
        for c, s in zip(chunks, scores, strict=True):
            if s >= 0:
                per_page[c.page_id].append(float(s))
        ranked = []
        for page, vals in per_page.items():
            vals.sort(reverse=True)
            ranked.append((page, vals[0] + 0.25 * (vals[1] if len(vals) > 1 else 0.0)))
        return sorted(ranked, key=lambda x: x[1], reverse=True)

    # ---- public API ------------------------------------------------------------
    def search(self, query: str, *, lang: str | None = None, top_k: int = 5) -> list[Hit]:
        s = self.settings
        scores = self.hybrid_scores(query, lang)
        top_pages = [p for p, _ in self.rank_pages(self.chunks, scores)[: s.max_pages]]
        if not top_pages:
            return []

        idx_by_page = defaultdict(list)
        for i, c in enumerate(self.chunks):
            if c.page_id in top_pages and scores[i] >= 0:
                idx_by_page[c.page_id].append(i)
        for page in idx_by_page:
            idx_by_page[page].sort(key=lambda i: scores[i], reverse=True)

        pool_size = max(top_k * 3, s.min_chunks_from_top_page)
        best_page = top_pages[0]
        pool = idx_by_page[best_page][: s.min_chunks_from_top_page]
        rest = sorted(
            (i for p in top_pages for i in idx_by_page[p] if i not in pool),
            key=lambda i: scores[i],
            reverse=True,
        )
        pool += rest[: pool_size - len(pool)]

        if self.reranker and pool:
            ranked = self.reranker.rerank(
                query, [self.chunks[i].embed_text() for i in pool], top_n=min(top_k, len(pool))
            )
            return [Hit(self.chunks[pool[j]], score, float(scores[pool[j]])) for j, score in ranked]
        pool.sort(key=lambda i: scores[i], reverse=True)
        return [Hit(self.chunks[i], float(scores[i]), float(scores[i])) for i in pool[:top_k]]

    def outline(self, page_id: str) -> list[tuple[str, str]]:
        seen, out = set(), []
        for c in self.by_page.get(page_id, []):
            if c.section_id not in seen:
                seen.add(c.section_id)
                out.append((c.section_id, c.section_title))
        return out

    def section_text(self, section_id: str) -> str | None:
        chunks = self.by_section.get(section_id)
        if not chunks:
            return None
        # Chunks overlap by design; for reading, the first chunk plus each later chunk is close enough.
        return "\n\n".join(c.text for c in chunks)
