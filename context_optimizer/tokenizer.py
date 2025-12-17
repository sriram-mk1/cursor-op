import tiktoken

def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Estimate the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))
