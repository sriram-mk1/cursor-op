# Context Optimizer Gateway

OpenAI-compatible LLM gateway that automatically optimizes context before forwarding requests to OpenRouter. Reduces token usage through intelligent context compression using BM25 retrieval, SimHash deduplication, and role-aware shrinking. Everything runs in-memory with no external dependencies.

## Features

- **Drop-in Replacement**: Works with Cursor, VS Code, Continue, and any OpenAI-compatible client
- **Automatic Context Optimization**: Intelligently compresses long conversation histories
- **Simple Setup**: Just set base URL and your OpenRouter API key
- **Streaming Support**: Full support for streaming responses
- **Zero External Dependencies**: No databases, embeddings, or vector stores required
- **All OpenRouter Models**: Access to 200+ models through one gateway

## Quickstart

### For AI Editors (Cursor, VS Code, etc.)

**Live Gateway**: `https://cursor-op.onrender.com`

1. Open your AI editor settings
2. Set custom base URL: `https://cursor-op.onrender.com`
3. Set API key: Your OpenRouter API key (`sk-or-v1-...`)
4. That's it! Context optimization is automatic.

### Using with cURL

```bash
curl -X POST https://cursor-op.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-or-v1-your-openrouter-key" \
  -d '{
    "model": "google/gemini-2.0-flash-lite-001",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## API Endpoints

### `POST /v1/chat/completions`
OpenAI-compatible chat completions with automatic context optimization.

**Headers:**
- `Authorization: Bearer <your-openrouter-key>` (required)

**Body Parameters:**
- `model`: OpenRouter model ID (e.g., `google/gemini-2.0-flash-lite-001`)
- `messages`: Array of message objects
- `enable_optimization`: Enable context optimization (default: `true`)
- `target_token_budget`: Max tokens for optimized context
- `max_chunks`: Max context chunks to retain (default: 12)
- All standard OpenAI parameters supported (`temperature`, `max_tokens`, `stream`, etc.)

### `GET /v1/models`
List available OpenRouter models.

### `GET /health`
Health check endpoint.

## Python Client Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://cursor-op.onrender.com/v1",
    api_key="sk-or-v1-your-openrouter-key"  # Your OpenRouter API key
)

response = client.chat.completions.create(
    model="google/gemini-2.0-flash-lite-001",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### Context Optimization (Automatic)

Context optimization is enabled by default for conversations with 4+ messages:

```python
# Long conversation - automatically optimized!
response = client.chat.completions.create(
    model="google/gemini-2.0-flash-lite-001",
    messages=[
        {"role": "user", "content": "My favorite color is blue."},
        {"role": "assistant", "content": "That's nice!"},
        {"role": "user", "content": "I love Python programming."},
        {"role": "assistant", "content": "Great choice!"},
        {"role": "user", "content": "What was my favorite color?"}
    ]
)
# Token savings: 40-70% on long conversations!
```

## Fixtures & Benchmarks

- `python scripts/generate_fixtures.py` writes `fixtures/stress_fixture.json` with long chat, logs, docs, and code snippets for stress. Use those artifacts in your agents to replay heavy history.
- `python scripts/benchmark.py` loads `fixtures/synthetic_fixture.json` (auto-generated) and prints ingestion + optimization latency plus raw vs. optimized token estimates.

## Testing

Run `pytest` for chunking, BM25 retrieval, SimHash dedup, and shrinker coverage.

## How It Works

1. **Ingestion**: Conversation history is chunked and indexed using BM25
2. **Optimization**: When messages exceed threshold, the optimizer:
   - Retrieves most relevant chunks using BM25 scoring
   - Deduplicates content using SimHash
   - Applies role-aware compression
   - Constructs optimized context within token budget
3. **Forwarding**: Optimized request is sent to OpenRouter
4. **Response**: LLM response is returned to client

## Deployment

### Live Instance

Already deployed at: `https://cursor-op.onrender.com`

### Deploy Your Own

1. Fork the repo
2. Create new Web Service on Render
3. Connect your GitHub repo
4. Deploy! (No environment variables needed)

## Token Savings

Run benchmarks to see token reduction:
```bash
python scripts/benchmark.py
```

Typical savings: 40-70% reduction in input tokens for long conversations.
