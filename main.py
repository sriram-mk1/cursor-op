import os
import time
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from context_optimizer import ContextOptimizer
from context_optimizer.engine import init_embedder


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
# LIFESPAN & OPTIMIZER
# ============================================================================

optimizer = ContextOptimizer(max_context_chunks=10)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    log.info("🚀 Starting Gateway...")
    # Pre-load the embedding model to avoid first-request latency
    init_embedder()
    yield
    log.info("🛑 Shutting down Gateway...")


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
    
    # Our params
    enable_optimization: Optional[bool] = True
    debug: Optional[bool] = False


# ============================================================================
# APP
# ============================================================================

app = FastAPI(
    title="Context Optimizer Gateway",
    version="3.2.0",
    description="Optimized OpenRouter proxy with RAG-based context optimization",
    lifespan=lifespan
)

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def extract_api_key(authorization: str) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    auth = authorization.strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return auth


async def fetch_and_log_analytics(generation_id: str, api_key: str):
    """Background task to fetch real analytics without blocking the user."""
    if not generation_id:
        return
        
    try:
        # Wait a tiny bit for OpenRouter to process the generation
        await asyncio.sleep(1.0)
        
        async with httpx.AsyncClient(timeout=10.0) as client:
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
                
                log.info(f"📊 REAL USAGE (Async): {model} | {total:,} tokens | ${cost:.6f}")
    except Exception as e:
        log.debug(f"Could not fetch async analytics: {e}")


@app.post("/v1/chat/completions")
@app.post("/api/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None, alias="authorization"),
    http_referer: Optional[str] = Header(None, alias="http-referer"),
    x_title: Optional[str] = Header(None, alias="x-title"),
):
    """Main endpoint with context optimization."""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    try:
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
        request_dict.pop("debug", None)
        request_dict["messages"] = messages
        request_dict.pop("prompt", None)
        
        processing_ms = (time.time() - start_time) * 1000
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": http_referer or "https://context-optimizer.app",
            "X-Title": x_title or "Context Optimizer Gateway"
        }
        
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
                    # Queue analytics fetch in background
                    generation_id = response_data.get("id")
                    if generation_id:
                        background_tasks.add_task(fetch_and_log_analytics, generation_id, api_key, original_count, len(messages), processing_ms)
                
                # Add our headers
                response_headers = {
                    "X-Context-Request-Id": request_id,
                    "X-Context-Original-Msgs": str(original_count),
                    "X-Context-Optimized-Msgs": str(len(messages)),
                    "X-Context-Processing-Ms": f"{processing_ms:.1f}",
                }
                
                # If debug is enabled, include reconstructed messages in the response
                if request.debug:
                    response_data["_debug"] = {
                        "reconstructed_messages": messages,
                        "original_count": original_count,
                        "optimized_count": len(messages)
                    }
                
                return JSONResponse(
                    content=response_data,
                    status_code=response.status_code,
                    headers=response_headers
                )
    except Exception as e:
        log.error(f"💥 Server Error: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Internal Gateway Error: {str(e)}")


async def fetch_and_log_analytics(generation_id: str, api_key: str, orig_msgs: int, opt_msgs: int, proc_ms: float):
    """Background task to fetch real analytics and log a single line summary."""
    if not generation_id: return
        
    try:
        # Wait for OpenRouter to process
        await asyncio.sleep(1.5)
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            usage_response = await client.get(
                f"{OPENROUTER_API_BASE}/generation",
                params={"id": generation_id},
                headers={"Authorization": f"Bearer {api_key}"}
            )
            
            if usage_response.status_code == 200:
                usage_data = usage_response.json().get("data", {})
                total = usage_data.get("tokens_prompt", 0) + usage_data.get("tokens_completion", 0)
                cost = usage_data.get("total_cost", 0)
                model = usage_data.get("model", "unknown").split("/")[-1]
                
                # THE ONE-LINE SUMMARY
                log.info(f"✅ [REQ] {orig_msgs}->{opt_msgs} msgs | {proc_ms:.0f}ms | {model} | {total:,} tkn | ${cost:.6f}")
    except Exception as e:
        log.debug(f"Async analytics failed: {e}")


# ============================================================================
# PASSTHROUGH ENDPOINTS
# ============================================================================

@app.get("/v1/models")
@app.get("/api/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    try:
        api_key = extract_api_key(authorization)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{OPENROUTER_API_BASE}/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            return JSONResponse(content=response.json(), status_code=response.status_code)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models/{model_id:path}")
@app.get("/api/v1/models/{model_id:path}")
async def get_model(model_id: str, authorization: Optional[str] = Header(None)):
    try:
        api_key = extract_api_key(authorization)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{OPENROUTER_API_BASE}/models/{model_id}",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            return JSONResponse(content=response.json(), status_code=response.status_code)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    try:
        api_key = extract_api_key(authorization)
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {"date": date} if date else {}
            response = await client.get(
                f"{OPENROUTER_API_BASE}/activity",
                params=params,
                headers={"Authorization": f"Bearer {api_key}"}
            )
            return JSONResponse(content=response.json(), status_code=response.status_code)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HEALTH & INFO
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "3.2.0",
        "uptime_seconds": int(time.time() - app.state.start_time) if hasattr(app.state, 'start_time') else 0,
        "features": ["smart_chunking", "bm25", "semantic_embeddings", "caching", "async_analytics"]
    }


@app.get("/")
async def root():
    return {
        "service": "Context Optimizer Gateway",
        "version": "3.2.0",
        "optimization": {
            "chunking": "Smart (target 600 chars)",
            "retrieval": "Hybrid (BM25 + Semantic)",
            "min_score": 0.4,
            "caching": "LRU (1000 items)"
        }
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    app.state.start_time = time.time()
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, workers=1)

