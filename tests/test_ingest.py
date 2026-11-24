from docs_agent.ingest import clean_markdown, page_id_for, page_url, split_sections

PAGE = """# Path Parameters { #path-parameters }

Intro text.

{* ../../docs_src/path_params/tutorial001.py hl[6:7] *}

## With <dfn title="types">types</dfn> { #with-types }

Declare `item_id: int`.

```python
# not a heading, inside code
x = 1
```

/// tip

Use Enum.

///
"""


def test_page_ids_and_urls():
    assert page_id_for("docs/ja/docs/tutorial/security/first-steps.md") == ("ja", "ja/tutorial/security/first-steps")
    assert page_id_for("docs/en/docs/tutorial/index.md") == ("en", "en/tutorial")
    assert page_url("en", "en/tutorial/cors") == "https://fastapi.tiangolo.com/tutorial/cors/"
    assert page_url("ja", "ja/tutorial/cors") == "https://fastapi.tiangolo.com/ja/tutorial/cors/"


def test_clean_removes_includes_admonitions_and_inline_tags():
    out = clean_markdown(PAGE)
    assert "{*" not in out and "///" not in out and "<dfn" not in out
    assert "Use Enum." in out


def test_split_sections_uses_headings_but_not_code_comments():
    sections = split_sections(PAGE, lang="en", page_id="en/tutorial/path-params")
    assert [s.section_title for s in sections] == ["Path Parameters", "With types"]
    assert all(s.page_title == "Path Parameters" for s in sections)
    assert "# not a heading" in sections[1].text
    assert sections[0].section_id == "en/tutorial/path-params#0"


def test_inline_code_replaces_references_and_drops_missing_files():
    from docs_agent.ingest import inline_code

    md = "Before\n\n{* ../../docs_src/status/tutorial001.py hl[6] *}\n\n{* ../../docs_src/missing.py *}\n\nAfter"
    out = inline_code(md, lambda p: "app.post('/items/', status_code=201)" if p.endswith("tutorial001.py") else None)
    assert "```python\napp.post('/items/', status_code=201)\n```" in out
    assert "missing" not in out and "{*" not in out
    assert out.startswith("Before") and out.endswith("After")
