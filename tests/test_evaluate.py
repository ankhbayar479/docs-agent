from docs_agent.agent import AgentResult
from docs_agent.evaluate import Case, abstained, evidence_found, facts_present, score, summarize


def test_fact_groups_need_every_group_with_any_alternative():
    facts = [["status_code"], ["201"]]
    assert facts_present("Set status_code=201.", facts)
    assert not facts_present("Set status_code.", facts)
    assert facts_present("It is kept in memory or on disk", [["memory", "disk"]])


def test_evidence_accepts_either_language_edition():
    r = AgentResult(question="q", answer="a", mode="agent", pages_seen=["ja/tutorial/cors"])
    assert evidence_found(r, ["tutorial/cors"])
    assert not evidence_found(r, ["tutorial/request-files"])


def test_abstain_detection_english_and_japanese():
    assert abstained("The tutorial does not cover Kubernetes autoscaling.")
    assert abstained("チュートリアルには記載がありません。")
    assert not abstained("Use the HorizontalPodAutoscaler with min 2 replicas.")


def test_score_and_summary():
    case = Case(id="x", lang="en", question="q", pages=["tutorial/cors"], facts=[["CORSMiddleware"]])
    good = AgentResult(question="q", answer="Add CORSMiddleware", mode="agent", pages_seen=["en/tutorial/cors"])
    bad = AgentResult(question="q", answer="Add a proxy", mode="baseline", pages_seen=["en/tutorial/cors"])
    scores = [score(case, good), score(case, bad)]
    assert [s.passed for s in scores] == [True, False]
    summary = summarize(scores)
    assert summary["agent"]["pass_rate"] == 100.0 and summary["baseline"]["pass_rate"] == 0.0


def test_abstain_detects_could_not_find():
    assert abstained("I'm sorry, I couldn't find any information about Kubernetes autoscaling.")
