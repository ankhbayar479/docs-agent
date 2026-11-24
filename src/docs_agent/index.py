"""Build and load the search index: chunks + BM25 keyword index + Cohere embeddings."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Settings
from .ingest import Section, load_corpus
from .text import chunk_text


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "en/tutorial/path-params#3.0"
    section_id: str
    page_id: str
    lang: str
    page_title: str
    section_title: str
    url: str
    text: str

    def embed_text(self) -> str:
        # Headings carry a lot of meaning in docs; put them in front of the body.
        return f"{self.page_title} > {self.section_title}\n\n{self.text}"


class Embedder(Protocol):
    def embed(self, texts: list[str], *, input_type: str) -> np.ndarray: ...


class CohereEmbedder:
    def __init__(self, client, model: str, batch_size: int = 96):
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: list[str], *, input_type: str) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self.client.embed(
                model=self.model,
                texts=texts[i : i + self.batch_size],
                input_type=input_type,  # "search_document" for chunks, "search_query" for questions
                embedding_types=["float"],
            )
            vectors.extend(resp.embeddings.float_)
        arr = np.asarray(vectors, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)


def make_chunks(sections: list[Section], settings: Settings) -> list[Chunk]:
    chunks = []
    for s in sections:
        for j, body in enumerate(chunk_text(s.text, max_chars=settings.chunk_chars, overlap=settings.chunk_overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{s.section_id}.{j}",
                    section_id=s.section_id,
                    page_id=s.page_id,
                    lang=s.lang,
                    page_title=s.page_title,
                    section_title=s.section_title,
                    url=s.url,
                    text=body,
                )
            )
    return chunks


def build_index(embedder: Embedder, settings: Settings | None = None) -> Path:
    settings = settings or Settings()
    sections = load_corpus(settings.corpus_path)
    chunks = make_chunks(sections, settings)
    started = time.perf_counter()
    vectors = embedder.embed([c.embed_text() for c in chunks], input_type="search_document")
    out = settings.index_dir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vectors)
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    meta = {
        "embed_model": getattr(embedder, "model", "unknown"),
        "fastapi_tag": settings.fastapi_tag,
        "chunks": len(chunks),
        "seconds": round(time.perf_counter() - started, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Indexed {len(chunks)} chunks in {meta['seconds']} s -> {out}")
    return out


def load_index(settings: Settings) -> tuple[list[Chunk], np.ndarray, dict]:
    d = settings.index_dir
    with (d / "chunks.jsonl").open(encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
    vectors = np.load(d / "embeddings.npy")
    meta = json.loads((d / "meta.json").read_text())
    if len(chunks) != len(vectors):
        raise SystemExit("Index is inconsistent; run `docs-agent index` again.")
    return chunks, vectors, meta
