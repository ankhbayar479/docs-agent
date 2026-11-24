import numpy as np

from docs_agent.retrieval import Retriever
from tests.conftest import FakeEmbedder


def make(corpus, settings, reranker=None):
    emb = FakeEmbedder()
    vectors = emb.embed([c.embed_text() for c in corpus], input_type="search_document")
    return Retriever(corpus, vectors, emb, settings, reranker=reranker)


def test_best_page_comes_first(corpus, settings):
    hits = make(corpus, settings).search("status 201", top_k=3)
    assert hits[0].chunk.page_id == "en/tutorial/response-status-code"


def test_two_stage_keeps_minimum_chunks_from_top_page(corpus, settings):
    hits = make(corpus, settings).search("status 201", top_k=3)
    top_page = [h for h in hits if h.chunk.page_id == "en/tutorial/response-status-code"]
    assert len(top_page) >= settings.min_chunks_from_top_page


def test_language_filter(corpus, settings):
    hits = make(corpus, settings).search("cors origin", lang="ja", top_k=5)
    assert hits and all(h.chunk.lang == "ja" for h in hits)


def test_only_top_pages_are_candidates(corpus, settings):
    hits = make(corpus, settings).search("status 201", top_k=10)
    assert len({h.chunk.page_id for h in hits}) <= settings.max_pages


def test_reranker_decides_final_order(corpus, settings):
    class Reverse:
        def rerank(self, query, documents, top_n):
            n = len(documents)
            return [(i, float(i) / n) for i in reversed(range(n))][:top_n]

    hits = make(corpus, settings, reranker=Reverse()).search("status 201", top_k=2)
    assert [round(h.score, 3) for h in hits] == sorted((round(h.score, 3) for h in hits), reverse=True)


def test_rank_pages_rewards_second_match():
    from tests.conftest import chunk

    chunks = [chunk("a", 0, 0, "x"), chunk("a", 1, 0, "x"), chunk("b", 0, 0, "x")]
    ranked = Retriever.rank_pages(chunks, np.array([0.8, 0.8, 0.9], dtype=np.float32))
    assert ranked[0][0] == "en/tutorial/a"  # 0.8 + 0.25*0.8 = 1.0 beats 0.9


def test_outline_and_section_text(corpus, settings):
    r = make(corpus, settings)
    assert r.outline("en/tutorial/response-status-code") == [
        ("en/tutorial/response-status-code#0", "section 0"),
        ("en/tutorial/response-status-code#1", "section 1"),
    ]
    assert "CORSMiddleware" in r.section_text("en/tutorial/cors#0")
    assert r.section_text("missing#0") is None
