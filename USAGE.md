# Usage Guide

## Gateway Information

- **Base URL**: `https://cursor-op.onrender.com`
- **Gateway API Key**: `dev-key-12345` (change in production!)
- **Status**: ✅ Live and operational

## Quick Start

### Using cURL

```bash
curl -X POST https://cursor-op.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-key-12345" \
  -H "X-OpenRouter-API-Key: YOUR_OPENROUTER_KEY" \
  -d '{
    "model": "google/gemini-2.0-flash-lite-001",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Using Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://cursor-op.onrender.com/v1",
    api_key="dev-key-12345",  # Gateway key
    default_headers={
        "X-OpenRouter-API-Key": "YOUR_OPENROUTER_KEY"
    }
)

response = client.chat.completions.create(
    model="google/gemini-2.0-flash-lite-001",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### Enable Context Optimization

```python
response = client.chat.completions.create(
    model="google/gemini-2.0-flash-lite-001",
    messages=[
        {"role": "user", "content": "My favorite color is blue."},
        {"role": "assistant", "content": "That's nice!"},
        {"role": "user", "content": "I love Python programming."},
        {"role": "assistant", "content": "Great choice!"},
        {"role": "user", "content": "What was my favorite color?"}
    ],
    extra_body={
        "enable_optimization": True,
        "max_chunks": 8,
        "target_token_budget": 100
    }
)
```

## Available Models

Any OpenRouter-compatible model, for example:
- `google/gemini-2.0-flash-lite-001`
- `anthropic/claude-3.5-sonnet`
- `openai/gpt-4-turbo`
- `meta-llama/llama-3.3-70b-instruct`

List all models:
```bash
curl https://cursor-op.onrender.com/v1/models \
  -H "Authorization: Bearer dev-key-12345" \
  -H "X-OpenRouter-API-Key: YOUR_OPENROUTER_KEY"
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info and usage |
| `/health` | GET | Health check |
| `/v1/chat/completions` | POST | Chat completions with optimization |
| `/v1/models` | GET | List available models |

## Configuration Options

### Context Optimization Parameters

- `enable_optimization` (boolean, default: `true`): Enable/disable context optimization
- `max_chunks` (int, default: `12`): Maximum number of context chunks to retain
- `target_token_budget` (int, optional): Target token budget for optimized context
- `user` (string, optional): User ID for session tracking (enables better optimization)

### Standard OpenAI Parameters

All standard OpenAI chat completion parameters are supported:
- `temperature`, `top_p`, `max_tokens`
- `stream`, `stop`
- `presence_penalty`, `frequency_penalty`
- `logit_bias`, `n`

## How Context Optimization Works

1. **Conversation history** is automatically analyzed
2. **BM25 retrieval** finds the most relevant past messages
3. **SimHash deduplication** removes redundant information
4. **Role-aware compression** shrinks context intelligently
5. **Optimized context** is sent to OpenRouter, saving tokens

### Benefits

- 🎯 **40-70% token reduction** on long conversations
- 🚀 **Faster responses** (less to process)
- 💰 **Lower costs** (fewer input tokens)
- 🧠 **Better performance** (focused context)

## Security

### Gateway Authentication

All requests require:
- `Authorization: Bearer <gateway-key>` header

### BYOK (Bring Your Own Key)

All requests require:
- `X-OpenRouter-API-Key: <your-openrouter-key>` header

The gateway never stores your OpenRouter API key.

## Testing

Run the test suite:
```bash
python test_gateway.py
```

## Production Deployment

### Change Gateway Key

1. Update environment variable on Render:
   ```
   GATEWAY_API_KEY=your-super-secure-key-here
   ```

2. Restart the service

### Monitor Usage

Check logs on Render Dashboard:
https://dashboard.render.com/web/srv-d50dm1npm1nc73elbosg

## Troubleshooting

### 401 Unauthorized
- Check your gateway API key in `Authorization` header

### 400 Bad Request - Missing X-OpenRouter-API-Key
- Add `X-OpenRouter-API-Key` header with your OpenRouter key

### 400 Bad Request - Invalid model ID
- Verify model name is correct
- List available models with `/v1/models` endpoint

### 500 Internal Server Error
- Check Render logs for details
- Service may be starting up (wait 30 seconds)

## Support

- **GitHub**: https://github.com/sriram-mk1/cursor-op
- **Render Dashboard**: https://dashboard.render.com/web/srv-d50dm1npm1nc73elbosg
