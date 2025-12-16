from context_optimizer import ContextOptimizer


def test_bm25_prefers_relevant_chunk():
    optimizer = ContextOptimizer()
    session_id = "bm25-session"
    optimizer.ingest(
        session_id,
        [
            {
                "role": "user",
                "source": "chat",
                "content": "We have an urgent bug in the payments module that needs a patch.",
            },
            {
                "role": "assistant",
                "source": "chat",
                "content": "I'll prioritize the payments bug and document the fix.",
            },
        ],
    )
    optimized = optimizer.optimize(session_id, query_text="payments bug", max_chunks=1)
    assert optimized["optimized_context"]
    summary = optimized["optimized_context"][0]["summary"].lower()
    assert "payments" in summary
    assert "bug" in summary
