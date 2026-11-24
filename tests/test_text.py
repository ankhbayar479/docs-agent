import itertools

from docs_agent.text import chunk_text, tokenize


def test_tokenize_english_words_and_japanese_bigrams():
    assert tokenize("Use status_code 201") == ["use", "status_code", "201"]
    assert tokenize("依存関係") == ["依存", "存関", "関係"]
    # Mixed text keeps both kinds of tokens.
    assert set(tokenize("CORSを設定")) >= {"cors", "を設", "設定"}


def test_short_text_is_one_chunk():
    assert chunk_text("short", max_chars=100, overlap=10) == ["short"]


def test_chunks_respect_max_and_overlap():
    paragraphs = "\n\n".join(f"paragraph {i} " + "x" * 80 for i in range(10))
    chunks = chunk_text(paragraphs, max_chars=300, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    # Each chunk after the first starts with the tail of the previous one.
    for prev, nxt in itertools.pairwise(chunks):
        assert nxt.startswith(prev[-40:])


def test_overlong_paragraph_is_hard_split():
    text = "".join(str(i % 10) for i in range(1000))
    chunks = chunk_text(text, max_chars=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    # Nothing is lost: every chunk (minus its overlap prefix) concatenates back to the original.
    rebuilt = chunks[0] + "".join(c.split("\n\n", 1)[1] for c in chunks[1:])
    assert rebuilt == text


def test_detect_lang():
    from docs_agent.text import detect_lang

    assert detect_lang("How do I return 201?") == "en"
    assert detect_lang("ステータスコード201を返すには？") == "ja"
    assert detect_lang("CORSMiddleware") == "en"
