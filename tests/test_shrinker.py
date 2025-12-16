from context_optimizer.shrinker import (
    shrink_authoritative,
    shrink_diagnostic,
    shrink_exploratory,
    shrink_historical,
)


def test_shrink_authoritative_focuses_query():
    content = "Line1\n// comment\nImportant detail: goal is to ship fast\nExtra context\nFinal note"
    summary = shrink_authoritative(content, {"goal"})
    assert "important detail" in summary.lower()


def test_shrink_diagnostic_removes_timestamps():
    content = "2024-01-01T12:00:00Z error occurred\nat function()"
    summary = shrink_diagnostic(content, set())
    assert "2024" not in summary
    assert "error" in summary


def test_shrink_exploratory_returns_title():
    content = "Title line\nDetails follow"
    summary = shrink_exploratory(content, set())
    assert summary.strip() == "Title line"


def test_shrink_historical_bullets_and_tail():
    content = "Old1\nOld2\nOld3\nLatest reply\nAssistant response"
    summary = shrink_historical(content, set())
    assert "•" in summary or "Latest reply" in summary
