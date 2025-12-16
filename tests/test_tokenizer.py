from context_optimizer.tokenizer import chunk_content, estimate_tokens, normalize_text


def test_chunk_content_breaks_long_text():
    text = " ".join(f"word{i}" for i in range(1200))
    chunks = chunk_content(text, target_tokens=300)
    assert len(chunks) >= 4
    for chunk in chunks:
        assert estimate_tokens(chunk) <= 320


def test_normalize_text_removes_stopwords():
    text = "This is a goal to test the optimizer."
    tokens = normalize_text(text)
    assert "goal" in tokens
    assert "the" not in tokens
