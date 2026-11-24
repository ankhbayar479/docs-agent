"""Shared fakes: no network, no API key."""

from __future__ import annotations

import numpy as np
import pytest

from docs_agent.config import Settings
from docs_agent.index import Chunk

VOCAB = ["status", "201", "cors", "origin", "upload", "file", "router", "yield"]


class FakeEmbedder:
    """Bag-of-words over a tiny vocabulary: deterministic and good enough to test ranking logic."""

    model = "fake-embed"

    def embed(self, texts, *, input_type):
        rows = []
        for t in texts:
            low = t.lower()
            v = np.array([float(low.count(w)) for w in VOCAB], dtype=np.float32) + 1e-3
            rows.append(v / np.linalg.norm(v))
        return np.vstack(rows)


def chunk(page: str, sec: int, part: int, text: str, lang: str = "en") -> Chunk:
    page_id = f"{lang}/tutorial/{page}"
    return Chunk(
        chunk_id=f"{page_id}#{sec}.{part}",
        section_id=f"{page_id}#{sec}",
        page_id=page_id,
        lang=lang,
        page_title=page,
        section_title=f"section {sec}",
        url=f"https://example.test/{page}/",
        text=text,
    )


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path)


@pytest.fixture
def corpus():
    return [
        chunk("response-status-code", 0, 0, "Use status_code=201 in the decorator. status 201 created."),
        chunk("response-status-code", 1, 0, "Status codes: 201 means created; use status names."),
        chunk("cors", 0, 0, "Add CORSMiddleware so another origin can call your API. cors origin."),
        chunk("request-files", 0, 0, "UploadFile stores large file uploads on disk. upload file."),
        chunk("bigger-applications", 0, 0, "Use APIRouter and include_router. router."),
        chunk("cors", 0, 0, "CORSMiddleware を追加すると別のオリジンから呼び出せます。cors origin", lang="ja"),
    ]
