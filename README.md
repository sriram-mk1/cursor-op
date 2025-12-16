# Context Optimizer Gateway

**OpenRouter-compatible API gateway with intelligent context optimization**. Reduces token usage by 40-70% through industry-standard techniques: BM25 retrieval, SimHash deduplication, and role-aware shrinking. Zero external dependencies - everything runs in-memory.

## Features

### 🚀 Full OpenRouter API Compatibility
- **All Parameters Supported**: temperature, top_p, top_k, min_p, top_a, frequency_penalty, presence_penalty, repetition_penalty, and more
- **Tool Calling**: Complete function calling support
- **Response Formatting**: JSON mode, structured outputs
- **Assistant Prefill**: Guide model responses
- **Streaming**: Full SSE streaming support
- **Model Routing**: Fallback routing and provider preferences
- **200+ Models**: Access to entire OpenRouter catalog

### 🧠 Intelligent Context Optimization
- **BM25 Retrieval**: Rank context chunks by relevance to current query
- **SimHash Deduplication**: Remove redundant/duplicate content
- **Role-Aware Shrinking**: Compress based on content type (code, logs, chat, docs)
- **Token Budget Enforcement**: Stay within specified token limits
- **Query-Based Caching**: Fast repeated optimizations
- **Automatic Activation**: Triggers on 4+ messages

### ⚡ Simple & Fast
- **Drop-in Replacement**: Works with Cursor, VS Code, Continue, and any OpenAI-compatible client
- **Zero Config**: No databases, no vector stores, no embeddings
- **In-Memory**: Everything runs locally with no external dependencies

## Quickstart

### For AI Editors (Cursor, VS Code, etc.)

**Live Gateway**: `https://cursor-op.onrender.com`

1. Open your AI editor settings
2. Set custom base URL: `https://cursor-op.onrender.com`
3. Set API key: Your OpenRouter API key (`sk-or-v1-...`)
4. That's it! Context optimization happens automatically.

### Basic Example

```bash
curl -X POST https://cursor-op.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-or-v1-your-openrouter-key" \
  -d '{
    "model": "anthropic/claude-3.5-sonnet",
    "messages": [
      {"role": "user", "content": "What is BM25?"}
    ]
  }'
```

### Advanced Example (All OpenRouter Features)

```bash
curl -X POST https://cursor-op.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-or-v1-your-key" \
  -H "HTTP-Referer: https://myapp.com" \
  -H "X-Title: My App" \
  -d '{
    "model": "openai/gpt-4o",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 0.9,
    "frequency_penalty": 0.5,
    "response_format": {"type": "json_object"},
    "stream": true,
    "enable_optimization": true,
    "max_chunks": 15,
    "target_token_budget": 8000
  }'
```

### Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py  # Starts on port 8000
```

## API Endpoints

### `POST /v1/chat/completions`
OpenRouter-compatible chat completions with intelligent context optimization.

**Headers:**
- `Authorization: Bearer <your-openrouter-key>` (required)
- `HTTP-Referer: <your-site-url>` (optional, for OpenRouter rankings)
- `X-Title: <your-app-name>` (optional, for OpenRouter rankings)

**OpenRouter Parameters (All Supported):**

**Core:**
- `model`: Model ID (e.g., `anthropic/claude-3.5-sonnet`)
- `messages`: Array of message objects OR `prompt`: String prompt
- `max_tokens`: Max tokens to generate
- `stream`: Enable streaming (default: false)
- `stop`: Stop sequences (string or array)

**Sampling:**
- `temperature`: Randomness (0.0-2.0, default: 1.0)
- `top_p`: Nucleus sampling (0.0-1.0, default: 1.0)
- `top_k`: Top-K sampling (integer)
- `top_a`: Top-A sampling (0.0-1.0)
- `min_p`: Minimum probability (0.0-1.0)

**Penalties:**
- `frequency_penalty`: Token frequency penalty (-2.0 to 2.0)
- `presence_penalty`: Token presence penalty (-2.0 to 2.0)
- `repetition_penalty`: Repetition penalty (0.0-2.0)

**Advanced:**
- `seed`: Deterministic sampling (integer)
- `logit_bias`: Token bias map
- `logprobs`: Return log probabilities (boolean)
- `top_logprobs`: Number of top logprobs (integer)
- `response_format`: Output format (e.g., `{"type": "json_object"}`)
- `prediction`: Predicted output for latency optimization

**Tool Calling:**
- `tools`: Array of tool definitions
- `tool_choice`: Tool selection strategy
- `parallel_tool_calls`: Enable parallel function calls (boolean)

**OpenRouter-Specific:**
- `transforms`: Message transforms
- `models`: Model fallback list
- `route`: Routing strategy
- `provider`: Provider preferences
- `user`: End-user identifier

**Context Optimization (Custom):**
- `enable_optimization`: Enable optimization (default: true)
- `target_token_budget`: Max tokens for context (integer)
- `max_chunks`: Max context chunks (default: 12)

### `GET /v1/models`
List available OpenRouter models.

### `GET /api/v1/generation?id=<generation_id>`
Query generation stats by ID (token counts, cost, etc.).

### `GET /health`
Health check endpoint.

## Python Client Examples

### Basic Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://cursor-op.onrender.com/v1",
    api_key="sk-or-v1-your-openrouter-key"
)

response = client.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",
    messages=[
        {"role": "user", "content": "Explain machine learning in simple terms"}
    ]
)

print(response.choices[0].message.content)
```

### Automatic Context Optimization

Context optimization triggers automatically for 4+ messages, saving 40-70% tokens:

```python
# Long conversation - automatically optimized!
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[
        {"role": "user", "content": "My favorite color is blue."},
        {"role": "assistant", "content": "That's nice!"},
        {"role": "user", "content": "I love Python programming."},
        {"role": "assistant", "content": "Great choice!"},
        {"role": "user", "content": "I'm learning machine learning."},
        {"role": "assistant", "content": "Exciting!"},
        {"role": "user", "content": "What was my favorite color?"}
    ]
)
```

### Advanced Features

```python
# JSON mode with streaming
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[
        {"role": "system", "content": "You return valid JSON only."},
        {"role": "user", "content": "List 3 programming languages"}
    ],
    response_format={"type": "json_object"},
    temperature=0.7,
    max_tokens=500,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Tool Calling (Function Calling)

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    tools=tools,
    tool_choice="auto"
)

print(response.choices[0].message.tool_calls)
```

### Custom Optimization Settings

```python
# Fine-tune optimization behavior
response = client.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",
    messages=long_conversation,
    extra_body={
        "enable_optimization": True,
        "target_token_budget": 16000,  # Max context tokens
        "max_chunks": 20                # Max chunks to retain
    }
)
```

## Fixtures & Benchmarks

- `python scripts/generate_fixtures.py` writes `fixtures/stress_fixture.json` with long chat, logs, docs, and code snippets for stress. Use those artifacts in your agents to replay heavy history.
- `python scripts/benchmark.py` loads `fixtures/synthetic_fixture.json` (auto-generated) and prints ingestion + optimization latency plus raw vs. optimized token estimates.

## Testing

Run `pytest` for chunking, BM25 retrieval, SimHash dedup, and shrinker coverage.

## How It Works

### Context Optimization Pipeline

When a request has 4+ messages, the gateway automatically:

1. **Chunking & Indexing**
   - Splits messages into semantic chunks
   - Builds BM25 inverted index for term frequency analysis
   - Computes SimHash fingerprints for each chunk

2. **Relevance Retrieval (BM25)**
   - Extracts query terms from latest user message
   - Scores all chunks using BM25 (industry-standard IR algorithm)
   - Ranks chunks by relevance to current query
   - Formula: `score = Σ(IDF(term) × (TF × (k1 + 1)) / (TF + k1 × (1 - b + b × doc_len/avg_len)))`

3. **Deduplication (SimHash)**
   - Compares SimHash fingerprints of top-ranked chunks
   - Removes near-duplicate content (Hamming distance ≤ 3)
   - Keeps most relevant unique chunks

4. **Role-Aware Compression**
   - **Authoritative** (system prompts, goals): Extract query-relevant lines
   - **Diagnostic** (logs, errors): Keep error messages + stack traces
   - **Exploratory** (code, general): First line or first 200 chars
   - **Historical** (documents): Bullet points + recent context

5. **Token Budget Enforcement**
   - Estimates tokens using fast tokenizer
   - Trims chunks to fit within `target_token_budget`
   - Prioritizes highest-scored chunks

6. **Request Forwarding**
   - Injects optimized context as system message
   - Forwards to OpenRouter with all original parameters
   - Returns response with optimization stats in headers

### Optimization Stats

Check response headers:
- `X-Context-Optimization: enabled`
- `X-Context-Original-Messages: 47`
- `X-Context-Optimized-Messages: 2`
- `X-Context-Token-Savings: 68.3%`

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
