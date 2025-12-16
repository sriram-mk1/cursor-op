import os
import time
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from context_optimizer import ContextOptimizer


class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Context optimization params
    enable_optimization: Optional[bool] = True
    target_token_budget: Optional[int] = None
    max_chunks: Optional[int] = 12


app = FastAPI(
    title="Context Optimizer Gateway",
    version="0.3.0",
    description="OpenAI-compatible gateway with automatic context optimization. Drop-in replacement for AI editors."
)
optimizer = ContextOptimizer()

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def extract_api_key(authorization: str = Header(None)):
    """Extract OpenRouter API key from Authorization header"""
    if not authorization:
        raise HTTPException(
            status_code=401, 
            detail="Missing Authorization header. Provide your OpenRouter API key as: Authorization: Bearer sk-or-v1-..."
        )
    
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization scheme. Use: Authorization: Bearer <api-key>")
    
    if not token:
        raise HTTPException(status_code=401, detail="Missing API key in Authorization header")
    
    return token


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: str = Header(None),
):
    """
    OpenAI-compatible chat completions endpoint with context optimization.
    
    Provide your OpenRouter API key via standard Authorization header:
    Authorization: Bearer sk-or-v1-...
    
    Works seamlessly with AI editors (Cursor, VS Code, etc.) by setting:
    - Base URL: https://cursor-op.onrender.com
    - API Key: Your OpenRouter API key
    """
    # Extract OpenRouter API key
    openrouter_api_key = extract_api_key(authorization)
    
    # Extract messages
    messages = [msg.dict() for msg in request.messages]
    
    # Optimize context if enabled
    if request.enable_optimization and len(messages) > 3:
        # Generate session ID from user or use timestamp
        session_id = request.user or f"session_{int(time.time())}"
        
        # Ingest conversation history (all messages except the last one)
        events = [
            {
                "role": msg["role"],
                "source": "chat",
                "content": msg["content"],
                "ts": time.time()
            }
            for msg in messages[:-1]
        ]
        
        if events:
            optimizer.ingest(session_id, events)
        
        # Optimize using the latest message as query
        latest_message = messages[-1]["content"]
        optimization_result = optimizer.optimize(
            session_id,
            latest_message,
            max_chunks=request.max_chunks or 12,
            target_token_budget=request.target_token_budget,
            cache_ttl_sec=60
        )
        
        # Reconstruct messages with optimized context
        if optimization_result.get("optimized_chunks"):
            optimized_context = "\n\n".join([
                chunk["content"] for chunk in optimization_result["optimized_chunks"]
            ])
            
            # Create new message list with system context + latest message
            messages = [
                {"role": "system", "content": f"Previous conversation context:\n{optimized_context}"},
                messages[-1]  # Latest user message
            ]
    
    # Prepare request for OpenRouter
    openrouter_request = {
        "model": request.model,
        "messages": messages,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "n": request.n,
        "stream": request.stream,
        "max_tokens": request.max_tokens,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
    }
    
    if request.stop:
        openrouter_request["stop"] = request.stop
    if request.logit_bias:
        openrouter_request["logit_bias"] = request.logit_bias
    
    # Call OpenRouter
    async with httpx.AsyncClient(timeout=60.0) as client:
        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://cursor-op.onrender.com",
            "X-Title": "Context Optimizer Gateway"
        }
        
        if request.stream:
            # Handle streaming
            async def generate():
                async with client.stream(
                    "POST",
                    f"{OPENROUTER_API_BASE}/chat/completions",
                    json=openrouter_request,
                    headers=headers
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"OpenRouter error: {error_text.decode()}"
                        )
                    
                    async for chunk in response.aiter_bytes():
                        yield chunk
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            # Handle non-streaming
            response = await client.post(
                f"{OPENROUTER_API_BASE}/chat/completions",
                json=openrouter_request,
                headers=headers
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"OpenRouter error: {response.text}"
                )
            
            return response.json()


@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    """Proxy OpenRouter models list"""
    openrouter_api_key = extract_api_key(authorization)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{OPENROUTER_API_BASE}/models",
            headers={"Authorization": f"Bearer {openrouter_api_key}"}
        )
        return response.json()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "context-optimizer-gateway", "version": "0.3.0"}


@app.get("/")
async def root():
    """Root endpoint with usage instructions"""
    return {
        "service": "Context Optimizer Gateway",
        "version": "0.3.0",
        "description": "OpenAI-compatible gateway with automatic context optimization. Drop-in replacement for AI editors.",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        },
        "setup": {
            "base_url": "https://cursor-op.onrender.com",
            "api_key": "Your OpenRouter API key (sk-or-v1-...)",
            "compatibility": "Works with Cursor, VS Code, Continue, and any OpenAI-compatible client"
        },
        "usage": {
            "authentication": "Standard Authorization: Bearer <your-openrouter-key>",
            "optimization": "Automatic context optimization (disable with enable_optimization: false)",
            "models": "All OpenRouter models supported"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
