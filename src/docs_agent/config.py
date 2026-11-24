"""Settings, read once from the environment (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # Source documents: FastAPI's tutorial, English and Japanese, pinned to one release.
    fastapi_tag: str = field(default_factory=lambda: os.getenv("FASTAPI_TAG", "0.165.0"))
    languages: tuple[str, ...] = ("en", "ja")
    doc_section: str = "tutorial"

    # Cohere models. Override in .env if Cohere renames or retires one.
    chat_model: str = field(default_factory=lambda: os.getenv("COHERE_CHAT_MODEL", "command-a-03-2025"))
    embed_model: str = field(default_factory=lambda: os.getenv("COHERE_EMBED_MODEL", "embed-v4.0"))
    rerank_model: str = field(default_factory=lambda: os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"))

    # Retrieval
    chunk_chars: int = 1200
    chunk_overlap: int = 150
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    max_pages: int = 3
    min_chunks_from_top_page: int = 2

    # Agent
    max_steps: int = 6
    max_tool_calls: int = 8  # a model can request several tools in one step; cap the total too
    temperature: float = 0.2
    # Cohere's newer Command models "think" before replying by default: ~12 s instead of ~1 s per call.
    thinking: str = field(default_factory=lambda: os.getenv("COHERE_THINKING", "disabled"))

    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DOCS_AGENT_DATA", PROJECT_ROOT / "data")))

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / "corpus.jsonl"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


def cohere_api_key() -> str:
    key = os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY")
    if not key:
        raise SystemExit("COHERE_API_KEY is not set. Copy .env.example to .env and add your key.")
    return key
