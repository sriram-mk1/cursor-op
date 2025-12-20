import os
import time
import logging
import asyncio
import json
import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from contextlib import asynccontextmanager

from dotenv import load_dotenv
# Load environment variables IMMEDIATELY
load_dotenv()

import httpx
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from context_optimizer.engine import ContextOptimizer, ENCODER
from database import Database

# OpenRouter Generation Endpoint
OPENROUTER_GEN_URL = "https://openrouter.ai/api/v1/generation"

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
# Initialized after load_dotenv()
from context_optimizer.engine import ContextOptimizer, ENCODER
from database import Database

optimizer = ContextOptimizer()
db = Database()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, api_key: str):
        await websocket.accept()
        if api_key not in self.active_connections:
            self.active_connections[api_key] = set()
        self.active_connections[api_key].add(websocket)
        log.info(f"WS Connected: {api_key[:8]}... Total: {len(self.active_connections[api_key])}")

    def disconnect(self, websocket: WebSocket, api_key: str):
        if api_key in self.active_connections:
            self.active_connections[api_key].remove(websocket)
            log.info(f"WS Disconnected: {api_key[:8]}...")

    async def broadcast(self, api_key: str, message: dict):
        if api_key in self.active_connections:
            log.info(f"Broadcasting to {len(self.active_connections[api_key])} clients for key {api_key[:8]}...")
            dead_connections = set()
            for connection in self.active_connections[api_key]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    log.error(f"Broadcast Error: {e}")
                    dead_connections.add(connection)
            for dead in dead_connections:
                self.active_connections[api_key].remove(dead)
        else:
            log.warning(f"No active WS connections for key {api_key[:8]}...")

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.start_time = time.time()
    log.info("🚀 V1 Session Gateway starting...")
    yield
    log.info("🛑 Gateway stopped")

app = FastAPI(title="V1 Session Gateway", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

class ChatRequest(BaseModel):
    model_config = {"extra": "allow"}
    messages: Optional[List[Any]] = None
    prompt: Optional[str] = None
    model: Optional[str] = "gpt-3.5-turbo"
    stream: Optional[bool] = False
    enable_optimization: Optional[bool] = True
    session_id: Optional[str] = None # Explicit session ID
    api_key: Optional[str] = None # Support key in payload

@app.get("/api/stats")
async def get_stats(x_v1_key: Optional[str] = Header(None, alias="x-v1-key")):
    if not x_v1_key:
        raise HTTPException(status_code=401, detail="Missing V1 API Key")
    stats = db.get_stats(x_v1_key)
    if not stats:
        raise HTTPException(status_code=401, detail="Invalid V1 API Key")
    return stats

@app.get("/api/user/stats")
async def get_user_stats(x_user_id: str = Header(..., alias="x-user-id")):
    return db.get_user_stats(x_user_id)

@app.get("/api/keys")
async def list_keys(x_user_id: str = Header(..., alias="x-user-id")):
    return db.list_keys(x_user_id)

class CreateKeyRequest(BaseModel):
    name: str
    user_id: str

@app.post("/api/keys")
async def create_key(req: CreateKeyRequest):
    return db.create_key(req.name, req.user_id)

@app.delete("/api/keys/{key}")
async def delete_key(key: str, x_user_id: str = Header(..., alias="x-user-id")):
    db.delete_key(key, x_user_id)
    return {"status": "deleted"}

class UpdateProviderKeyRequest(BaseModel):
    v1_key: str
    user_id: str
    openrouter_key: str

@app.post("/api/provider-key")
async def update_provider_key(req: UpdateProviderKeyRequest):
    db.update_provider_key(req.v1_key, req.user_id, req.openrouter_key)
    return {"status": "updated"}

@app.websocket("/ws/{api_key}")
async def websocket_endpoint(websocket: WebSocket, api_key: str):
    log.info(f"Incoming WS connection for key: {api_key[:8]}...")
    # Validate key
    key_data = db.validate_key(api_key)
    if not key_data:
        log.warning(f"Invalid WS key attempt: {api_key}")
        await websocket.close(code=4003)
        return
    
    await manager.connect(websocket, api_key)
    try:
        # Initial stats
        stats = db.get_stats(api_key)
        await websocket.send_json({"type": "init", "data": stats})
        while True:
            await websocket.receive_text() # Keep alive
    except WebSocketDisconnect:
        manager.disconnect(websocket, api_key)
    except Exception as e:
        log.error(f"WS Error: {e}")
        manager.disconnect(websocket, api_key)

@app.post("/v1/chat/completions")
@app.post("/api/v1/chat/completions")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    authorization: Optional[str] = Header(None, alias="authorization"),
    x_v1_key: Optional[str] = Header(None, alias="x-v1-key"),
    x_session_id: Optional[str] = Header(None, alias="x-session-id"),
):
    start_time = time.time()
    # 1. Resolve V1 Gateway Key
    v1_key = x_v1_key or (authorization.replace("Bearer ", "").strip() if authorization and "v1-" in authorization else None)
    key_data = db.validate_key(v1_key) if v1_key else None

    # 2. Resolve OpenRouter API Key
    or_key = request.api_key # Priority 1: Payload
    
    # Priority 2: Authorization Header (if it's a standard SK key)
    if not or_key and authorization and "v1-" not in authorization:
        or_key = authorization.replace("Bearer ", "").strip()
    
    # Priority 3: Associated Provider Key from Database
    if not or_key and key_data and key_data.get("openrouter_key"):
        or_key = key_data.get("openrouter_key")
        
    # Priority 4: Environment Fallback
    if not or_key or or_key.strip() == "":
        or_key = os.getenv("OPENROUTER_API_KEY")

    if not or_key:
        raise HTTPException(status_code=401, detail="Missing OpenRouter API Key. Provide it in payload, Authorization header, or link it to your V1 key in the dashboard.")

    # 2. Session Tracking & History Sync
    session_id = request.session_id or x_session_id
    msgs = request.messages or []
    if request.prompt and not msgs:
        msgs = [{"role": "user", "content": request.prompt}]

    if not session_id and msgs:
        key_id = v1_key[:8] if v1_key else "anon"
        first_content = str(msgs[0].get("content", ""))
        fingerprint = hashlib.md5(f"{key_id}:{first_content[:500]}".encode()).hexdigest()[:12]
        session_id = f"s_{key_id}_{fingerprint}"

    # --- Ghost Session Logic (Supabase Powered) ---
    # We maintain the "True History" in Supabase to understand context growth
    db_state = db.get_session_state(session_id) or {"history": []}
    raw_history = db_state["history"]
    
    # Sync new turns
    if msgs:
        if not raw_history or msgs[0].get("content") != raw_history[0].get("content"):
            raw_history = msgs
        else:
            # Sync only new turns
            if len(msgs) > len(raw_history):
                for m in msgs[len(raw_history):]:
                    raw_history.append(m)
    
    original_tokens = sum(len(ENCODER.encode(json.dumps(m))) for m in raw_history)
    
    # 3. Context Window Discovery (Observability Phase)
    # We pass the full raw_history to the optimizer, but it might just be 
    # logging for now if optimization is disabled.
    optimized_msgs = msgs
    reconstruction_log = {}

    if request.enable_optimization and len(raw_history) > 1 and key_data:
        try:
            user_key_id = v1_key or "anon"
            optimized_msgs, reconstruction_log = optimizer.optimize(user_key_id, session_id, raw_history)
            if reconstruction_log and "total_history_tokens" in reconstruction_log:
                original_tokens = reconstruction_log["total_history_tokens"]
        except Exception as e:
            log.error(f"Optimization Error: {e}")

    # 4. Forward
    payload = request.model_dump(exclude_none=True)
    for k in ["enable_optimization", "session_id", "api_key", "debug", "prompt"]:
        payload.pop(k, None)
    payload["messages"] = optimized_msgs

    headers = {"Authorization": f"Bearer {or_key}", "Content-Type": "application/json"}

    async def log_and_broadcast(gen_id: str, status_code: int, latency_ms: float, response_msg: Dict = None):
        if not v1_key: return
        
        # Async Metadata
        or_metadata = None
        if gen_id and status_code == 200:
            async with httpx.AsyncClient(timeout=10.0) as client:
                for i in range(3):
                    await asyncio.sleep(1.5 * (i + 1))
                    try:
                        resp = await client.get(OPENROUTER_GEN_URL, params={"id": gen_id}, headers={"Authorization": f"Bearer {or_key}"})
                        if resp.status_code == 200:
                            or_metadata = resp.json().get("data")
                            if or_metadata and or_metadata.get("tokens_prompt"): break
                    except: pass

        try:
            # Save the new response message to history
            if status_code == 200 and response_msg:
                raw_history.append(response_msg)
                db.save_session_state(session_id, {"history": raw_history})

            log_entry = db.log_request(
                v1_key, session_id, request.model, 
                original_tokens, latency_ms,
                reconstruction_log, or_metadata,
                raw_messages=raw_history,
                response_message=response_msg
            )
            await manager.broadcast(v1_key, {"type": "request", "data": log_entry})
        except Exception as e:
            log.error(f"Log Task Error: {e}")

    if request.stream:
        async def stream_gen():
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", f"{OPENROUTER_API_BASE}/chat/completions", json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        yield f"data: {json.dumps({'error': {'message': 'OpenRouter Error'}})}\n\n".encode()
                        return
                    
                    full_content = ""
                    gen_id = None
                    async for line in resp.aiter_lines():
                        yield (line + "\n").encode()
                        
                        if line.startswith("data: "):
                            content = line[6:].strip()
                            if content == "[DONE]": continue
                            try:
                                d = json.loads(content)
                                if not gen_id: gen_id = d.get("id")
                                delta = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    full_content += delta
                            except: pass
                    
                    background_tasks.add_task(log_and_broadcast, gen_id, 200, (time.time() - start_time) * 1000, {"role": "assistant", "content": full_content})
        return StreamingResponse(stream_gen(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{OPENROUTER_API_BASE}/chat/completions", json=payload, headers=headers)
            resp_json = resp.json()
            resp_msg = resp_json.get("choices", [{}])[0].get("message")
            background_tasks.add_task(log_and_broadcast, resp_json.get("id"), resp.status_code, (time.time() - start_time) * 1000, resp_msg)
            return JSONResponse(content=resp_json, status_code=resp.status_code)

@app.get("/health")
async def health():
    return {"status": "ok", "uptime": int(time.time() - app.state.start_time)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
