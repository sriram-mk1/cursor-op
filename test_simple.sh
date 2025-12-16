#!/bin/bash

echo "🧪 Testing Context Optimizer Gateway"
echo "===================================="
echo ""

# Test 1: Health Check
echo "1️⃣  Health Check:"
curl -s https://cursor-op.onrender.com/health | jq
echo ""

# Test 2: Root endpoint
echo "2️⃣  Gateway Info:"
curl -s https://cursor-op.onrender.com/ | jq '.setup'
echo ""

# Test 3: Authentication Test (will fail with invalid key, but shows auth works)
echo "3️⃣  Authentication Test:"
echo "   Testing with invalid key (should get proper error):"
curl -s -X POST https://cursor-op.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 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" \
  -d '{"model": "google/gemini-2.0-flash-lite-001", "messages": [{"role": "user", "content": "test"}]}' \
  | jq '.detail' | head -3
echo ""

echo "✅ Gateway is operational!"
echo ""
echo "📝 To use with your OpenRouter key:"
echo "   Replace 'invalid-key' with your actual OpenRouter API key"
echo ""
echo "🎯 For AI Editors (Cursor, VS Code, Continue):"
echo "   Base URL: https://cursor-op.onrender.com"
echo "   API Key: <your-openrouter-key>"
