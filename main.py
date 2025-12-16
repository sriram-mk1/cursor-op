import os
import time
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

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


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
@app.api_route("/api/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_all_routes(path: str, request: Request):
    """
    Catch-all proxy for all OpenRouter API routes.
    Only applies context optimization to /v1/chat/completions POST requests.
    Everything else is passed through transparently.
    """
    # Extract API key
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Provide your OpenRouter API key as: Authorization: Bearer sk-or-v1-..."
        )
    
    # Get request body if present
    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.json()
        except:
            body = await request.body()
    
    # Apply context optimization ONLY to chat completions
    if path == "chat/completions" and request.method == "POST" and isinstance(body, dict):
        # Convert to ChatCompletionRequest
        try:
            req = ChatCompletionRequest(**body)
            
            # Extract messages
            messages = [msg.dict() for msg in req.messages]
            
            # Optimize context if enabled and enough messages
            if req.enable_optimization and len(messages) > 3:
                session_id = req.user or f"session_{int(time.time())}"
                
                # Ingest conversation history
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
                    max_chunks=req.max_chunks or 12,
                    target_token_budget=req.target_token_budget,
                    cache_ttl_sec=60
                )
                
                # Reconstruct messages with optimized context
                if optimization_result.get("optimized_chunks"):
                    optimized_context = "\n\n".join([
                        chunk["content"] for chunk in optimization_result["optimized_chunks"]
                    ])
                    
                    messages = [
                        {"role": "system", "content": f"Previous conversation context:\n{optimized_context}"},
                        messages[-1]
                    ]
            
            # Update body with potentially optimized messages
            body["messages"] = messages
        except Exception as e:
            # If optimization fails, pass through original request
            pass
    
    # Proxy to OpenRouter
    openrouter_url = f"{OPENROUTER_API_BASE}/{path}"
    
    # Prepare headers
    headers = {
        "Authorization": auth_header,
        "Content-Type": request.headers.get("content-type", "application/json"),
        "HTTP-Referer": request.headers.get("referer", "https://cursor-op.onrender.com"),
        "X-Title": "Context Optimizer Gateway"
    }
    
    # Add any other relevant headers
    for key, value in request.headers.items():
        if key.lower() not in ["host", "content-length", "connection", "authorization", "content-type"]:
            headers[key] = value
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Handle streaming
        if body and isinstance(body, dict) and body.get("stream"):
            async def generate():
                async with client.stream(
                    request.method,
                    openrouter_url,
                    json=body if isinstance(body, dict) else None,
                    content=body if isinstance(body, bytes) else None,
                    headers=headers,
                    params=dict(request.query_params)
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
        
        # Handle non-streaming
        else:
            response = await client.request(
                method=request.method,
                url=openrouter_url,
                json=body if isinstance(body, dict) else None,
                content=body if isinstance(body, bytes) else None,
                headers=headers,
                params=dict(request.query_params)
            )
            
            # Return response with same status code and headers
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type")
            )


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
        "description": "Full transparent proxy for OpenRouter with automatic context optimization.",
        "endpoints": {
            "all_openrouter_routes": "/v1/* or /api/v1/*",
            "health": "/health"
        },
        "setup": {
            "base_url": "https://cursor-op.onrender.com",
            "api_key": "Your OpenRouter API key (sk-or-v1-...)",
            "compatibility": "Works with Cursor, VS Code, Continue, and any OpenAI-compatible client"
        },
        "features": {
            "transparent_proxy": "All OpenRouter API routes supported",
            "context_optimization": "Automatic for /v1/chat/completions with 4+ messages",
            "streaming": "Full support for streaming responses"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
