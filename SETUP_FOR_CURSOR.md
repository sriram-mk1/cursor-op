# Setup for Cursor (and other AI Editors)

## Quick Setup

1. Open Cursor Settings (⌘ + ,)
2. Go to **Models** section
3. Click **Add Model** or configure OpenAI override
4. Set these values:
   - **Base URL**: `https://cursor-op.onrender.com/v1`
   - **API Key**: Your OpenRouter API key (get one at https://openrouter.ai/keys)

## What You Get

✅ **Automatic Context Optimization**: Long conversations are automatically compressed, saving tokens and costs

✅ **All OpenRouter Models**: Access to 200+ models including:
- Claude (Anthropic)
- GPT-4 (OpenAI)
- Gemini (Google)
- Llama (Meta)
- And many more!

✅ **Same Response Format**: Works exactly like OpenAI - no changes needed to your workflow

## Example Models to Use

```
google/gemini-2.0-flash-lite-001  (fast, cheap)
anthropic/claude-3.5-sonnet       (best for coding)
openai/gpt-4-turbo                (powerful)
meta-llama/llama-3.3-70b-instruct (open source)
```

## How It Works

The gateway automatically:
1. Detects long conversations (4+ messages)
2. Applies BM25 retrieval to find relevant context
3. Deduplicates with SimHash
4. Compresses intelligently
5. Sends optimized request to OpenRouter
6. Returns exact OpenRouter response

**Token Savings**: 40-70% reduction on long conversations!

## Tracking

The gateway adds optional headers you can inspect:
- `X-Context-Optimization`: Shows if optimization was applied
- `X-Original-Message-Count`: Original number of messages
- `X-Optimized-Message-Count`: After optimization

## Troubleshooting

### Authentication Error
- Make sure you're using a valid OpenRouter API key
- Get your key at: https://openrouter.ai/keys
- Format should be: `sk-or-v1-...`

### Models Not Showing
- The gateway supports all OpenRouter models
- Check available models at: https://openrouter.ai/models

### Want to Disable Optimization?
Add `"enable_optimization": false` to your requests (advanced users only)

## Cost Savings Example

**Without optimization** (100 message conversation):
- Input: 50,000 tokens @ $0.15/1M = $7.50
- Output: 5,000 tokens @ $0.60/1M = $3.00
- **Total: $10.50**

**With optimization** (same conversation):
- Input: 15,000 tokens @ $0.15/1M = $2.25
- Output: 5,000 tokens @ $0.60/1M = $3.00
- **Total: $5.25** ✅ **Save $5.25 (50%)**

## Support

- GitHub: https://github.com/sriram-mk1/cursor-op
- Issues: https://github.com/sriram-mk1/cursor-op/issues

## Privacy

- Your API key is never stored
- Requests are proxied directly to OpenRouter
- No logging of sensitive data
- Open source - verify the code yourself!
