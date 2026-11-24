import copy
import json
from types import SimpleNamespace as NS

from docs_agent.agent import DocsAgent
from docs_agent.retrieval import Retriever
from docs_agent.tools import ToolBox
from tests.conftest import FakeEmbedder


def usage(i, o):
    return NS(tokens=NS(input_tokens=i, output_tokens=o))


def tool_call(cid, name, **args):
    return NS(id=cid, type="function", function=NS(name=name, arguments=json.dumps(args)))


def answer(text):
    return NS(message=NS(tool_calls=None, tool_plan=None, content=[NS(type="text", text=text)]), usage=usage(10, 5))


class ScriptedClient:
    """Plays back a fixed list of responses and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def chat(self, **kwargs):
        # Snapshot: the agent keeps appending to the same messages list after the call.
        self.requests.append(copy.deepcopy(kwargs))
        return self.responses.pop(0)


def build(corpus, settings, client):
    emb = FakeEmbedder()
    vectors = emb.embed([c.embed_text() for c in corpus], input_type="search_document")
    return DocsAgent(client, ToolBox(Retriever(corpus, vectors, emb, settings)), settings)


def test_agent_searches_reads_then_answers(corpus, settings):
    client = ScriptedClient(
        [
            NS(
                message=NS(
                    tool_calls=[tool_call("c1", "search_docs", query="status 201")],
                    tool_plan="Search first.",
                    content=None,
                ),
                usage=usage(100, 20),
            ),
            NS(
                message=NS(
                    tool_calls=[tool_call("c2", "read_section", section_id="en/tutorial/response-status-code#0")],
                    tool_plan="Read the section.",
                    content=None,
                ),
                usage=usage(200, 20),
            ),
            answer("Use status_code=201 in the decorator."),
        ]
    )
    result = build(corpus, settings, client).ask("How do I return 201?")
    assert result.answer == "Use status_code=201 in the decorator."
    assert [s.tool for s in result.steps] == ["search_docs", "read_section"]
    assert result.pages_seen[0] == "en/tutorial/response-status-code"
    assert (result.input_tokens, result.output_tokens) == (310, 45)
    # Tool results go back as Cohere tool messages with a document payload.
    tool_msg = client.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert json.loads(tool_msg["content"][0]["document"]["data"])["results"]


def test_agent_stops_after_max_steps_and_answers_without_tools(corpus, settings):
    looping = [
        NS(
            message=NS(tool_calls=[tool_call(f"c{i}", "search_docs", query="cors")], tool_plan="", content=None),
            usage=usage(1, 1),
        )
        for i in range(settings.max_steps)
    ]
    client = ScriptedClient([*looping, answer("Add CORSMiddleware.")])
    result = build(corpus, settings, client).ask("cors?")
    assert result.stopped_early and result.answer == "Add CORSMiddleware."
    assert "tools" not in client.requests[-1]


def test_bad_tool_arguments_become_an_error_message(corpus, settings):
    client = ScriptedClient(
        [
            NS(
                message=NS(
                    tool_calls=[tool_call("c1", "read_section", section_id="nope#9")], tool_plan="", content=None
                ),
                usage=usage(1, 1),
            ),
            answer("Not covered."),
        ]
    )
    result = build(corpus, settings, client).ask("?")
    assert "unknown section_id" in result.steps[0].result_preview


def test_baseline_passes_documents_and_makes_one_call(corpus, settings):
    client = ScriptedClient([answer("Use CORSMiddleware.")])
    result = build(corpus, settings, client).ask_baseline("cors origin")
    assert result.mode == "baseline" and result.tool_calls == 1
    assert client.requests[0]["documents"] and "tools" not in client.requests[0]


def test_question_language_drives_search_default_and_answer_instruction(corpus, settings):
    client = ScriptedClient(
        [
            NS(
                message=NS(
                    tool_calls=[tool_call("c1", "search_docs", query="cors origin")], tool_plan="", content=None
                ),
                usage=usage(1, 1),
            ),
            answer("CORSMiddleware を追加します。"),
        ]
    )
    agent = build(corpus, settings, client)
    result = agent.ask("別のオリジンからAPIを呼ぶには？")
    assert agent.toolbox.default_lang == "ja"
    assert result.pages_seen == ["ja/tutorial/cors"]
    assert client.requests[0]["messages"][1]["content"].endswith("(Answer in Japanese.)")
    assert client.requests[0]["thinking"] == {"type": settings.thinking}


def test_read_section_with_a_page_id_returns_the_outline(corpus, settings):
    agent = build(corpus, settings, ScriptedClient([]))
    out = agent.toolbox.call("read_section", {"section_id": "en/tutorial/cors"})
    assert out["sections"][0]["section_id"] == "en/tutorial/cors#0"


def test_baseline_document_ids_are_unique_even_within_one_section(corpus, settings):
    from tests.conftest import chunk

    corpus = [*corpus, chunk("response-status-code", 0, 1, "More about status 201 status codes.")]
    client = ScriptedClient([answer("Use status_code=201.")])
    build(corpus, settings, client).ask_baseline("status 201", top_k=5)
    ids = [d["id"] for d in client.requests[0]["documents"]]
    assert len(ids) == len(set(ids))


def test_total_tool_calls_are_capped_even_when_requested_in_parallel(corpus, settings):
    burst = [tool_call(f"c{i}", "search_docs", query="cors") for i in range(20)]
    client = ScriptedClient(
        [NS(message=NS(tool_calls=burst, tool_plan="", content=None), usage=usage(1, 1)), answer("Add CORSMiddleware.")]
    )
    result = build(corpus, settings, client).ask("cors?")
    executed = [s for s in result.steps if "tool budget used up" not in s.result_preview]
    assert len(executed) == settings.max_tool_calls
    # Every requested call still gets a reply, so the conversation stays valid for the API.
    tool_msgs = [m for m in client.requests[-1]["messages"] if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_msgs) == 20
