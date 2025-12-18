import os
import time
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from context_optimizer.engine import ContextOptimizer

# Clean Logging
class ToonLog(logging.Formatter):
    def format(self, record):
        ts = datetime.now().strftime('%H:%M:%S')
        level = record.levelname[0]
        return f"\033[2m{ts}\033[0m [{level}] {record.getMessage()}"

log = logging.getLogger("gateway")
log.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(ToonLog())
log.addHandler(sh)

# App State
optimizer = ContextOptimizer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.start_time = time.time()
    log.info("🚀 V1 Session Gateway starting...")
    yield
    log.info("🛑 Gateway stopped")

app = FastAPI(title="V1 Session Gateway", lifespan=lifespan)
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

class ChatRequest(BaseModel):
    model_config = {"extra": "allow"}
    messages: Optional[List[Any]] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    stream: Optional[bool] = False
    enable_optimization: Optional[bool] = True
    session_id: Optional[str] = None # Explicit session ID

@app.post("/v1/chat/completions")
@app.post("/api/v1/chat/completions")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    authorization: Optional[str] = Header(None, alias="authorization"),
    x_session_id: Optional[str] = Header(None, alias="x-session-id"),
):
    api_key = authorization.replace("Bearer ", "").strip() if authorization else ""
    if not api_key: raise HTTPException(status_code=401, detail="Missing API Key")

    # Determine Session ID
    session_id = request.session_id or x_session_id or "default_session"
    
    msgs = request.messages or []
    if request.prompt and not msgs:
        msgs = [{"role": "user", "content": request.prompt}]
    
    # V1 Pipeline: Optimize with Session Context
    if request.enable_optimization and len(msgs) > 1:
        try:
            msgs = optimizer.optimize(session_id, msgs)
        except Exception as e:
            log.error(f"V1 Pipeline Error: {e}")

    # Forward
    payload = request.model_dump(exclude_none=True)
    payload.pop("enable_optimization", None)
    payload.pop("session_id", None)
    payload.pop("debug", None)
    payload["messages"] = msgs
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
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        log.error(f"Forward Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "uptime": int(time.time() - app.state.start_time)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
