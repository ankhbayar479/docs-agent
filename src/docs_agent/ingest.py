"""Download the FastAPI tutorial (EN + JA) and split it into sections.

Output: one JSON line per section with its page, heading, language and URL.
No API key needed.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from .config import Settings

TREE_URL = "https://api.github.com/repos/fastapi/fastapi/git/trees/{tag}?recursive=1"
RAW_URL = "https://raw.githubusercontent.com/fastapi/fastapi/{tag}/{path}"
SITE_URL = "https://fastapi.tiangolo.com"


@dataclass(frozen=True)
class Section:
    section_id: str  # "en/tutorial/path-params#2"
    page_id: str  # "en/tutorial/path-params"
    lang: str
    page_title: str
    section_title: str
    text: str
    url: str


def list_pages(client: httpx.Client, settings: Settings) -> list[str]:
    resp = client.get(TREE_URL.format(tag=settings.fastapi_tag))
    resp.raise_for_status()
    paths = []
    for item in resp.json()["tree"]:
        path = item["path"]
        for lang in settings.languages:
            prefix = f"docs/{lang}/docs/{settings.doc_section}/"
            if item["type"] == "blob" and path.startswith(prefix) and path.endswith(".md"):
                paths.append(path)
    return sorted(paths)


# FastAPI's docs embed code by reference ({* ../../docs_src/x.py *}) and use
# "/// note | Title" admonition fences. Neither carries prose worth indexing.
_CODE_INCLUDE = re.compile(r"\{\*.*?\*\}", re.S)
_ADMONITION = re.compile(r"^///.*$", re.M)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HEADING = re.compile(r"^(#{1,3})\s+(.*?)\s*(\{.*\})?\s*$")
# Inline glossary tags (<dfn title="...">word</dfn>, <abbr ...>) wrap words in prose and headings.
_INLINE_TAG = re.compile(r"</?(?:dfn|abbr|span|small)\b[^>]*>")


def clean_markdown(md: str) -> str:
    md = _HTML_COMMENT.sub("", md)
    md = _CODE_INCLUDE.sub("", md)
    md = _ADMONITION.sub("", md)
    md = _INLINE_TAG.sub("", md)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def page_id_for(path: str) -> tuple[str, str]:
    # docs/ja/docs/tutorial/security/first-steps.md -> ("ja", "ja/tutorial/security/first-steps")
    parts = path.split("/")
    lang = parts[1]
    rel = "/".join(parts[3:]).removesuffix(".md").removesuffix("/index")
    return lang, f"{lang}/{rel}"


def page_url(lang: str, page_id: str) -> str:
    rel = page_id.split("/", 1)[1]
    prefix = "" if lang == "en" else f"/{lang}"
    return f"{SITE_URL}{prefix}/{rel}/"


def split_sections(md: str, *, lang: str, page_id: str) -> list[Section]:
    """Split one page on its #, ## and ### headings. Text before the first heading joins the page intro."""
    lines = clean_markdown(md).splitlines()
    page_title = page_id.rsplit("/", 1)[-1]
    sections: list[tuple[str, list[str]]] = []
    current_title, buf = "", []
    in_code = False
    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else _HEADING.match(line)
        if m:
            if m.group(1) == "#" and not sections and not "".join(buf).strip():
                page_title = m.group(2)
                current_title = page_title
                continue
            if "".join(buf).strip() or current_title:
                sections.append((current_title or page_title, buf))
            current_title, buf = m.group(2), []
        else:
            buf.append(line)
    sections.append((current_title or page_title, buf))

    url = page_url(lang, page_id)
    out = []
    for i, (title, body) in enumerate(s for s in sections if "\n".join(s[1]).strip()):
        out.append(
            Section(
                section_id=f"{page_id}#{i}",
                page_id=page_id,
                lang=lang,
                page_title=page_title,
                section_title=title,
                text="\n".join(body).strip(),
                url=url,
            )
        )
    return out


def ingest(settings: Settings | None = None) -> Path:
    settings = settings or Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        paths = list_pages(client, settings)
        sections: list[Section] = []
        for path in paths:
            resp = client.get(RAW_URL.format(tag=settings.fastapi_tag, path=path))
            resp.raise_for_status()
            lang, page_id = page_id_for(path)
            sections.extend(split_sections(resp.text, lang=lang, page_id=page_id))
    with settings.corpus_path.open("w", encoding="utf-8") as f:
        for s in sections:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    pages = {s.page_id for s in sections}
    print(f"Ingested {len(paths)} files -> {len(pages)} pages, {len(sections)} sections -> {settings.corpus_path}")
    return settings.corpus_path


def load_corpus(path: Path) -> list[Section]:
    with path.open(encoding="utf-8") as f:
        return [Section(**json.loads(line)) for line in f if line.strip()]
