# Context Optimizer Gateway

OpenAI-compatible LLM gateway that automatically optimizes context before forwarding requests to OpenRouter. Reduces token usage through intelligent context compression using BM25 retrieval, SimHash deduplication, and role-aware shrinking. Everything runs in-memory with no external dependencies.

## Features

- **OpenAI-compatible API**: Drop-in replacement for OpenAI's chat completions endpoint
- **Automatic Context Optimization**: Intelligently compresses long conversation histories
- **BYOK (Bring Your Own Key)**: Uses your OpenRouter API key for model access
- **Streaming Support**: Full support for streaming responses
- **Zero External Dependencies**: No databases, embeddings, or vector stores required

## Quickstart

### Local Development

1. Clone and install:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Set environment variables:
```bash
cp .env.example .env
# Edit .env and set your GATEWAY_API_KEY
```

3. Run the server:
```bash
uvicorn main:app --reload
```

### Using the Gateway

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-gateway-key" \
  -H "X-OpenRouter-API-Key: your-openrouter-key" \
  -d '{
    "model": "anthropic/claude-3.5-sonnet",
    "messages": [{"role": "user", "content": "Hello!"}],
    "enable_optimization": true
  }'
```

## API Endpoints

### `POST /v1/chat/completions`
OpenAI-compatible chat completions with automatic context optimization.

**Headers:**
- `Authorization: Bearer <gateway-key>` (required)
- `X-OpenRouter-API-Key: <your-openrouter-key>` (required)

**Body Parameters:**
- `model`: OpenRouter model ID (e.g., `anthropic/claude-3.5-sonnet`)
- `messages`: Array of message objects
- `enable_optimization`: Enable context optimization (default: `true`)
- `target_token_budget`: Max tokens for optimized context
- `max_chunks`: Max context chunks to retain (default: 12)
- All standard OpenAI parameters supported

### `GET /v1/models`
List available OpenRouter models.

### `GET /health`
Health check endpoint.

## Python Client Example

```python
import openai

client = openai.OpenAI(
    base_url="https://your-gateway.onrender.com/v1",
    api_key="your-gateway-key",
    default_headers={
        "X-OpenRouter-API-Key": "your-openrouter-key"
    }
)

response = client.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",
    messages=[
        {"role": "user", "content": "Tell me about context optimization"}
    ],
    extra_body={"enable_optimization": True}
)

print(response.choices[0].message.content)
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

## Deployment to Render

The service is ready to deploy to Render:

1. Push to GitHub
2. Create new Web Service on Render
3. Set environment variable: `GATEWAY_API_KEY=<your-secure-key>`
4. Deploy!

## Token Savings

Run benchmarks to see token reduction:
```bash
python scripts/benchmark.py
```

Typical savings: 40-70% reduction in input tokens for long conversations.
