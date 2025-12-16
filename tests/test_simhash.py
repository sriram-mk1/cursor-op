from context_optimizer import ContextOptimizer


def test_simhash_deduplication():
    optimizer = ContextOptimizer()
    session_id = "simhash-session"
    payload = [
        {"role": "tool", "source": "log", "content": "error: disk full at 12:00:01"},
        {"role": "tool", "source": "log", "content": "error: disk full at 12:00:01"},
    ]
    first = optimizer.ingest(session_id, [payload[0]])
    second = optimizer.ingest(session_id, [payload[1]])
    assert first["ingested"] == 1
    assert second["ingested"] == 0
    assert second["deduped"] >= 1
