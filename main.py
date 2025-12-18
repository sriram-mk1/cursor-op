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

from context_optimizer.engine import ContextOptimizer, get_embedder


# ============================================================================
# LOGGING
# ============================================================================

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        level = record.levelname[0]
        msg = record.getMessage()
        color = "\033[32m" if record.levelno == logging.INFO else "\033[31m" if record.levelno >= logging.ERROR else "\033[36m"
        return f"\033[2m{ts}\033[0m {color}[{level}]\033[0m {msg}"

def setup_logging():
    logger = logging.getLogger("gateway")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    logger.addHandler(handler)
    return logger

log = setup_logging()


# ============================================================================
# APP & OPTIMIZER
# ============================================================================

optimizer = ContextOptimizer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.start_time = time.time()
    log.info("🚀 Gateway v5.0.0 (Hybrid RAG) starting...")
    # Pre-load embedder
    get_embedder()
    yield
    log.info("🛑 Gateway shutting down")

app = FastAPI(title="Context Optimizer", lifespan=lifespan)
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


# ============================================================================
# MODELS
# ============================================================================

class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}
    messages: Optional[List[Any]] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    stream: Optional[bool] = False
    enable_optimization: Optional[bool] = True


# ============================================================================
# HELPERS
# ============================================================================

def extract_api_key(authorization: str) -> str:
    if not authorization: return ""
    return authorization.replace("Bearer ", "").strip()

async def log_usage(generation_id: str, api_key: str, orig_count: int, opt_count: int):
    if not generation_id or not api_key: return
    try:
        await asyncio.sleep(2.0)
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{OPENROUTER_API_BASE}/generation",
                params={"id": generation_id},
                headers={"Authorization": f"Bearer {api_key}"}
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                tokens = data.get("tokens_prompt", 0) + data.get("tokens_completion", 0)
                cost = data.get("total_cost", 0)
                log.info(f"📊 {orig_count}->{opt_count} msgs | {tokens:,} tkn | ${cost:.6f}")
    except: pass


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.post("/v1/chat/completions")
@app.post("/api/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None, alias="authorization"),
):
    api_key = extract_api_key(authorization)
    if not api_key: raise HTTPException(status_code=401, detail="Missing API Key")

    messages = request.messages or []
    if request.prompt and not messages:
        messages = [{"role": "user", "content": request.prompt}]
    
    orig_count = len(messages)
    
    # Optimize
    try:
        if request.enable_optimization:
            messages = optimizer.optimize(messages)
    except Exception as e:
        log.error(f"Optimization failed: {e}")
        # Fallback to original messages if optimization fails
    
    opt_count = len(messages)

    payload = request.model_dump(exclude_none=True)
    payload.pop("enable_optimization", None)
    payload["messages"] = messages
    if "prompt" in payload: payload.pop("prompt")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        if request.stream:
            async def stream_gen():
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream("POST", f"{OPENROUTER_API_BASE}/chat/completions", json=payload, headers=headers) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
            return StreamingResponse(stream_gen(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{OPENROUTER_API_BASE}/chat/completions", json=payload, headers=headers)
                data = resp.json()
                if resp.status_code == 200:
                    gen_id = data.get("id")
                    background_tasks.add_task(log_usage, gen_id, api_key, orig_count, opt_count)
                return JSONResponse(content=data, status_code=resp.status_code)
    except Exception as e:
        log.error(f"Forwarding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "uptime": int(time.time() - app.state.start_time)}

@app.get("/")
async def root():
    return {"name": "Hybrid RAG Gateway", "version": "5.0.0"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
