# Context Optimizer Gateway

An OpenRouter-compatible API gateway with intelligent context optimization using semantic embeddings. Reduces LLM token usage by 40-90% while preserving conversation relevance.

## 🚀 Features

- **Semantic Retrieval**: Uses `sentence-transformers/static-retrieval-mrl-en-v1` (4MB) for efficient CPU-based similarity search
- **OpenRouter Compatible**: Drop-in replacement for OpenRouter API
- **Automatic Optimization**: Intelligently shrinks conversation history based on query relevance
- **Low Memory**: Optimized for 512MB RAM environments (Railway free tier)
- **Fast**: Sub-second optimization even for large conversations
- **Token Aware**: Accurate token counting with tiktoken

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🏃 Quick Start

### Local Development

```bash
python main.py
```

Server runs on `http://localhost:8000`

### Production Deployment (Railway)

1. Fork this repository
2. Create a new project on Railway
3. Connect your GitHub repository
4. Railway will automatically detect the `Dockerfile` and build the service
5. The service will be deployed automatically (using CPU-optimized PyTorch)




## 🔧 Usage

### As OpenRouter Replacement

Simply change your base URL:

```python
import openai

client = openai.OpenAI(
    base_url="https://cursor-op.up.railway.app/v1",
    api_key="YOUR_OPENROUTER_API_KEY"
)

response = client.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",
    messages=[
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "Tell me about Python"}
    ]
)
```

### Configuration Options

Add these parameters to your request:

```python
response = client.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",
    messages=[...],
    extra_body={
        "enable_optimization": True,      # Enable/disable (default: True)
        "max_chunks": 12,                 # Max relevant chunks to keep
        "target_token_budget": 2000       # Max tokens for context
    }
)
```

### cURL Example

```bash
curl -X POST https://cursor-op.up.railway.app/v1/chat/completions \
  -H "Authorization: Bearer YOUR_OPENROUTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-3.5-sonnet",
    "messages": [
      {"role": "user", "content": "What is FastAPI?"}
    ],
    "enable_optimization": true,
    "max_chunks": 8
  }'
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Request                          │
│  POST /v1/chat/completions                                  │
│  messages: [msg1, msg2, ..., msg20]                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Context Optimizer (Semantic Retrieval)           │
│                                                             │
│  1. Ingest: Store conversation history                     │
│  2. Encode: Generate embeddings with potion-base-2M        │
│  3. Retrieve: Find most relevant chunks via similarity     │
│  4. Shrink: Truncate long content (logs, files)           │
│  5. Budget: Enforce token limits                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Optimized Request                          │
│                                                             │
│  messages: [                                                │
│    {role: "system", content: "[Optimized context]"},      │
│    {role: "user", content: "Current query"}               │
│  ]                                                          │
│                                                             │
│  Token reduction: 40-90%                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Forward to OpenRouter                      │
│  - Lower token costs                                       │
│  - Faster responses                                        │
│  - Preserved relevance                                     │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Performance

### Token Reduction
- **Small conversations** (3-10 messages): 40-60% reduction
- **Medium conversations** (10-30 messages): 60-80% reduction  
- **Large conversations** (30+ messages): 80-90% reduction

### Speed
- **Optimization time**: <1 second for most conversations
- **Memory usage**: ~100-200MB (fits in 512MB Render free tier)

### Example Results
```
Original: 2,901 tokens → Optimized: 1,349 tokens
Reduction: 53.5% | Time: 0.8s
```

## 🛠️ How It Works

1. **Ingestion**: Conversation history is stored in-memory per session
2. **Embedding**: Query and chunks are encoded using lightweight static embeddings
3. **Retrieval**: Cosine similarity ranks chunks by relevance to current query
4. **Selection**: Top N chunks are selected within token budget
5. **Injection**: Optimized context is injected into system prompt
6. **Forwarding**: Reduced payload is sent to OpenRouter

## 🔍 API Endpoints

### Chat Completions
- `POST /v1/chat/completions`
- `POST /api/v1/chat/completions`

Full OpenRouter API compatibility with additional optimization parameters.

### Models
- `GET /v1/models` - List available models
- `GET /v1/models/{model_id}` - Get model details

### Health
- `GET /health` - Service health check
- `GET /` - API documentation

## 📝 Response Headers

When optimization is enabled, responses include:

```
X-Context-Optimization: enabled
X-Context-Original-Messages: 20
X-Context-Optimized-Messages: 2
X-Context-Token-Savings: 75.3%
```

## 🔐 Environment Variables

```bash
PORT=8000                    # Server port (default: 8000)
```

Your OpenRouter API key is passed via the `Authorization` header in each request.

## 🧪 Testing

The optimizer automatically logs detailed metrics:

```
--- OPTIMIZATION START for Session xyz ---
Query: How do I fix the error?
Total available chunks: 15
Used static embeddings for retrieval.
Optimized: 8 chunks, ~450 tokens
Savings: 68.2%
--- OPTIMIZATION END ---
```

## 🚦 When Optimization Triggers

- **Enabled**: Conversations with 4+ messages
- **Disabled**: Short conversations (≤3 messages) pass through unchanged
- **Override**: Set `enable_optimization: false` to disable per-request

## 💡 Best Practices

1. **Session IDs**: Use consistent user IDs for better context management
2. **Token Budgets**: Set `target_token_budget` based on your model's context window
3. **Max Chunks**: Adjust `max_chunks` (default: 12) based on conversation complexity
4. **Monitoring**: Check response headers to track optimization effectiveness

## 🔧 Configuration

### Embedding Model
The system uses `sentence-transformers/static-retrieval-mrl-en-v1` by default. To change:

```python
# context_optimizer/engine.py
self.model = SentenceTransformer("your-model-name", device="cpu")
```

### Memory Optimization
- Model is loaded once at startup
- Sessions are stored in-memory (automatic cleanup)
- Maximum sequence length: 256 tokens (configurable)

## 📚 Tech Stack

- **FastAPI**: High-performance async web framework
- **sentence-transformers**: Semantic embedding generation
- **tiktoken**: Accurate GPT token counting
- **httpx**: Async HTTP client for OpenRouter

## 🤝 Contributing

This is a production-ready service. For issues or improvements:

1. Check logs for optimization metrics
2. Adjust `max_chunks` and `target_token_budget` parameters
3. Monitor response headers for effectiveness

## 📄 License

MIT License - Use freely in your projects

## 🙏 Acknowledgments

- Built with [sentence-transformers](https://www.sbert.net/)
- Powered by [OpenRouter](https://openrouter.ai/)
- Embedding model: [sentence-transformers/static-retrieval-mrl-en-v1](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1)

---

**Status**: 🟢 Production Ready | **Memory**: ~100-200MB | **Speed**: <1s optimization
