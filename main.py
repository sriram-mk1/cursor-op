"""
Context Optimizer Gateway
=========================
OpenRouter-compatible proxy with intelligent context optimization.

Features:
- Smart chunking (respects code blocks, logs)
- Hybrid retrieval (BM25 + semantic embeddings)
- Real analytics from OpenRouter
"""

import os
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from context_optimizer import ContextOptimizer


# ============================================================================
# LOGGING
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Clean, colorful logging."""
    
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'DIM': '\033[2m',
    }
    
    def format(self, record):
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        dim = self.COLORS['DIM']
        bold = self.COLORS['BOLD']
        level = record.levelname[0]
        return f"{dim}{ts}{reset} {color}{bold}[{level}]{reset} {record.getMessage()}"


def setup_logging():
    logger = logging.getLogger("gateway")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    logger.addHandler(handler)
    return logger


log = setup_logging()


# ============================================================================
# OPTIMIZER
# ============================================================================

optimizer = ContextOptimizer(max_context_chunks=15)


# ============================================================================
# MODELS
# ============================================================================

class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}
    
    messages: Optional[List[Any]] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    transforms: Optional[List[str]] = None
    route: Optional[str] = None
    
    # Our param
    enable_optimization: Optional[bool] = True


# ============================================================================
# APP
# ============================================================================

app = FastAPI(
    title="Context Optimizer Gateway",
    version="3.0.0",
    description="OpenRouter proxy with RAG-based context optimization"
)

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def extract_api_key(authorization: str) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    auth = authorization.strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return auth


@app.post("/v1/chat/completions")
@app.post("/api/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None, alias="authorization"),
    http_referer: Optional[str] = Header(None, alias="http-referer"),
    x_title: Optional[str] = Header(None, alias="x-title"),
):
    """Main endpoint with context optimization."""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    api_key = extract_api_key(authorization)
    
    if not request.messages and not request.prompt:
        raise HTTPException(status_code=400, detail="Either 'messages' or 'prompt' is required")
    
    # Convert prompt to messages
    if request.prompt and not request.messages:
        request.messages = [{"role": "user", "content": request.prompt}]
    
    # Normalize messages
    messages = []
    for msg in request.messages:
        if isinstance(msg, dict):
            messages.append(msg)
        elif hasattr(msg, 'model_dump'):
            messages.append(msg.model_dump(exclude_none=True))
        elif hasattr(msg, 'dict'):
            messages.append(msg.dict(exclude_none=True))
        else:
            messages.append({"role": "user", "content": str(msg)})
    
    original_count = len(messages)
    
    # ========================================================================
    # CONTEXT OPTIMIZATION
    # ========================================================================
    if request.enable_optimization and len(messages) > 3:
        messages = optimizer.optimize(messages)
    
    # Build request
    request_dict = request.model_dump(exclude_none=True)
    request_dict.pop("enable_optimization", None)
    request_dict["messages"] = messages
    request_dict.pop("prompt", None)
    
    processing_ms = (time.time() - start_time) * 1000
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": http_referer or "https://context-optimizer.app",
        "X-Title": x_title or "Context Optimizer Gateway"
    }
    
    log.info(f"🚀 Forwarding to OpenRouter ({original_count} → {len(messages)} msgs, {processing_ms:.1f}ms)")
    
    if request.stream:
        async def generate():
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{OPENROUTER_API_BASE}/chat/completions",
                        json=request_dict,
                        headers=headers
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk
            except Exception as e:
                log.error(f"Streaming error: {e}")
                yield f'data: {{"error": "{str(e)}"}}\n\n'.encode()
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{OPENROUTER_API_BASE}/chat/completions",
                    json=request_dict,
                    headers=headers
                )
                
                response_data = response.json()
                
                if response.status_code != 200:
                    log.error(f"OpenRouter error {response.status_code}: {response.text[:200]}")
                else:
                    # ================================================================
                    # LOG REAL ANALYTICS FROM OPENROUTER
                    # ================================================================
                    generation_id = response_data.get("id")
                    
                    if generation_id:
                        try:
                            usage_response = await client.get(
                                f"{OPENROUTER_API_BASE}/generation",
                                params={"id": generation_id},
                                headers={"Authorization": f"Bearer {api_key}"}
                            )
                            
                            if usage_response.status_code == 200:
                                usage_data = usage_response.json().get("data", {})
                                
                                prompt_tokens = usage_data.get("tokens_prompt", 0)
                                completion_tokens = usage_data.get("tokens_completion", 0)
                                total = prompt_tokens + completion_tokens
                                cost = usage_data.get("total_cost", 0)
                                model = usage_data.get("model", "unknown")
                                
                                log.info(f"📊 REAL USAGE:")
                                log.info(f"   Model: {model}")
                                log.info(f"   Prompt: {prompt_tokens:,} tokens")
                                log.info(f"   Completion: {completion_tokens:,} tokens")
                                log.info(f"   Total: {total:,} tokens")
                                log.info(f"   Cost: ${cost:.6f}")
                        except Exception as e:
                            log.warning(f"Could not fetch analytics: {e}")
                    
                    # Also check inline usage
                    usage = response_data.get("usage", {})
                    if usage and not generation_id:
                        log.info(f"📈 Usage: prompt={usage.get('prompt_tokens', 0):,}, "
                                f"completion={usage.get('completion_tokens', 0):,}")
                
                return JSONResponse(
                    content=response_data,
                    status_code=response.status_code,
                    headers={
                        "X-Context-Request-Id": request_id,
                        "X-Context-Original-Msgs": str(original_count),
                        "X-Context-Optimized-Msgs": str(len(messages)),
                        "X-Context-Processing-Ms": f"{processing_ms:.1f}",
                    }
                )
        except httpx.HTTPError as e:
            log.error(f"OpenRouter request failed: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to reach OpenRouter: {str(e)}")


# ============================================================================
# PASSTHROUGH ENDPOINTS
# ============================================================================

@app.get("/v1/models")
@app.get("/api/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    api_key = extract_api_key(authorization)
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{OPENROUTER_API_BASE}/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return JSONResponse(content=response.json(), status_code=response.status_code)


@app.get("/v1/models/{model_id:path}")
@app.get("/api/v1/models/{model_id:path}")
async def get_model(model_id: str, authorization: Optional[str] = Header(None)):
    api_key = extract_api_key(authorization)
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{OPENROUTER_API_BASE}/models/{model_id}",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return JSONResponse(content=response.json(), status_code=response.status_code)


# ============================================================================
# ANALYTICS
# ============================================================================

@app.get("/activity")
@app.get("/v1/activity")
@app.get("/api/v1/activity")
async def get_activity(
    date: Optional[str] = None,
    authorization: Optional[str] = Header(None)
):
    """Get real usage analytics from OpenRouter."""
    api_key = extract_api_key(authorization)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        params = {"date": date} if date else {}
        response = await client.get(
            f"{OPENROUTER_API_BASE}/activity",
            params=params,
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return JSONResponse(content=response.json(), status_code=response.status_code)


# ============================================================================
# HEALTH & INFO
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "requests": optimizer.request_count,
        "features": ["smart_chunking", "bm25", "semantic_embeddings"]
    }


@app.get("/")
async def root():
    return {
        "service": "Context Optimizer Gateway",
        "version": "3.0.0",
        "optimization": {
            "chunking": "Smart (respects code blocks)",
            "retrieval": "Hybrid (BM25 + MiniLM-L6-v2)",
            "max_chunks": 15
        },
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "activity": "/activity",
            "health": "/health"
        }
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    
    log.info("=" * 60)
    log.info("🚀 Context Optimizer Gateway v3.0.0")
    log.info("=" * 60)
    log.info(f"📍 Port: {port}")
    log.info("📦 Features: Smart Chunking | BM25 | Semantic Embeddings")
    log.info("=" * 60)
    
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
