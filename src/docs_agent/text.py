"""Text utilities: chunking and a tokenizer that works for English and Japanese."""

from __future__ import annotations

import re

_LATIN = re.compile(r"[a-z0-9_]+")
# Hiragana, katakana, CJK ideographs, half-width katakana.
_CJK_RUN = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ]+")


def tokenize(text: str) -> list[str]:
    """Words for Latin script, overlapping character bigrams for Japanese.

    Japanese has no spaces, so whitespace tokenization would turn a whole sentence
    into one "word" and keyword search would never match. Bigrams are the usual
    dependency-free fix.
    """
    text = text.lower()
    tokens = _LATIN.findall(text)
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def chunk_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """Split on paragraph boundaries; hard-split paragraphs longer than max_chars.

    Consecutive chunks share up to `overlap` characters so a fact cut at a boundary
    still appears whole in one of them.
    """
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    # A piece may later be prefixed with `overlap` chars plus a blank line, so cap pieces
    # below max_chars by that much; otherwise the joined chunk would exceed max_chars.
    piece_max = max(1, max_chars - overlap - 2)
    pieces: list[str] = []
    for p in paragraphs:
        while len(p) > piece_max:
            pieces.append(p[:piece_max])
            p = p[piece_max:]
        pieces.append(p)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        current = f"{tail}\n\n{piece}" if tail else piece
    if current:
        chunks.append(current)
    return chunks
