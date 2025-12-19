import os
import time
import logging
import asyncio
import json
import hashlib
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
        # Send initial stats
        stats = db.get_stats(api_key)
        log.info(f"Sending initial stats to {api_key[:8]}...")
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
    # 1. Resolve OpenRouter Key (Primary)
    # Priority: 
    # 1. api_key in payload
    # 2. Authorization header (if it's a real OR key, not a V1 key)
    # 3. Environment variable OPENROUTER_API_KEY
    or_key = request.api_key
    
    if not or_key and authorization and "v1-" not in authorization:
        or_key = authorization.replace("Bearer ", "").strip()
    
    if not or_key or or_key.strip() == "":
        or_key = os.getenv("OPENROUTER_API_KEY")
    
    if not or_key:
        raise HTTPException(status_code=401, detail="Missing OpenRouter API Key. Please provide it in the Authorization header.")

    # 2. Resolve V1 Key for Tracking (Optional)
    v1_key = x_v1_key or (authorization.replace("Bearer ", "").strip() if authorization and "v1-" in authorization else None)
    key_data = db.validate_key(v1_key) if v1_key else None

    # 3. Determine Session ID (Smart Fingerprinting for External Editors)
    session_id = request.session_id or x_session_id
    
    msgs = request.messages or []
    if request.prompt and not msgs:
        msgs = [{"role": "user", "content": request.prompt}]

    if not session_id and msgs:
        # Create a stable fingerprint from the first message AND the user's key
        key_id = v1_key[:8] if v1_key else "anon"
        
        # V3.6 Denoised Fingerprinting: 
        # We strip the environment garbage BEFORE hashing so the session is "sticky"
        # regardless of timestamp/cost drift in the boilerplate.
        first_content = str(msgs[0].get("content", ""))
        # Simple extraction of the actual task/user input
        task_match = re.search(r"<task>(.*?)</task>", first_content, re.DOTALL | re.IGNORECASE)
        core_msg = task_match.group(1).strip() if task_match else first_content[:500]
        
        # Salt with the user's key for 100% isolation
        fingerprint = hashlib.md5(f"{key_id}:{core_msg}".encode()).hexdigest()[:12]
        session_id = f"t_{key_id}_{fingerprint}"
        log.info(f"🧩 Stable Fingerprint: {session_id} (Denoised)")
    elif session_id:
        log.info(f"📌 Using provided session: {session_id}")
    
    if not session_id:
        # Final fallback for purely stateless calls without conversation prefix
        session_id = f"s_{v1_key[:8]}" if v1_key else "global_stateless"
        log.info(f"⚡️ Using fallback: {session_id}")
    
    original_tokens = sum(len(ENCODER.encode(json.dumps(m))) for m in msgs)
    
    # 4. V1 Pipeline: Optimize Context (if enabled)
    optimized_msgs = msgs
    reconstruction_log = {}

    if request.enable_optimization and len(msgs) > 1 and key_data:
        try:
            # Use v1_key for strict isolation
            user_key_id = v1_key or "anon"
            log.info(f"🚀 [V3.3] Optimizing {session_id} for user {user_key_id[:8]}")
            optimized_msgs, reconstruction_log = optimizer.optimize(user_key_id, session_id, msgs)
            log.info(f"✅ [V3.3] Optimized | {reconstruction_log.get('selected_lines')} / {reconstruction_log.get('total_lines')} lines kept")
        except Exception as e:
            log.error(f"V1 Pipeline Error: {e}")

    optimized_tokens = sum(len(ENCODER.encode(json.dumps(m))) for m in optimized_msgs)
    tokens_saved = max(0, original_tokens - optimized_tokens)

    # 4. Forward to OpenRouter
    payload = request.model_dump(exclude_none=True)
    payload.pop("enable_optimization", None)
    payload.pop("session_id", None)
    payload.pop("api_key", None)
    payload.pop("debug", None)
    payload["messages"] = optimized_msgs
    if "prompt" in payload: payload.pop("prompt")

    headers = {"Authorization": f"Bearer {or_key}", "Content-Type": "application/json"}

    async def fetch_or_metadata(generation_id: str, or_key: str, max_retries=3):
        """Fetch accurate usage and cost from OpenRouter with retries."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            for i in range(max_retries):
                try:
                    # Give OpenRouter a moment to populate the data
                    await asyncio.sleep(1.5 * (i + 1)) 
                    
                    resp = await client.get(
                        OPENROUTER_GEN_URL,
                        params={"id": generation_id},
                        headers={"Authorization": f"Bearer {or_key}"}
                    )
                    
                    if resp.status_code == 200:
                        data = resp.json().get("data")
                        if data and data.get("tokens_prompt"):
                            return data
                    
                    log.warning(f"Retry {i+1} for generation {generation_id}: {resp.status_code}")
                except Exception as e:
                    log.error(f"Error fetching OR metadata: {e}")
        return None

    async def log_and_broadcast(gen_id: str, or_key: str, status_code: int, latency_ms: float):
        if not key_data or not v1_key:
            log.warning("Skipping broadcast: No key data or V1 key")
            return
            
        # Fetch accurate data from OpenRouter
        or_metadata = None
        if gen_id and status_code == 200:
            or_metadata = await fetch_or_metadata(gen_id, or_key)

        try:
            log.info(f"Logging request for {v1_key[:8]}... Status: {status_code} Latency: {latency_ms:.1f}ms")
            log_entry = db.log_request(
                v1_key, session_id, request.model, 
                original_tokens, latency_ms,
                reconstruction_log,
                or_metadata=or_metadata
            )
            log.info(f"Broadcasting log entry to {v1_key[:8]}...")
            await manager.broadcast(v1_key, {"type": "request", "data": log_entry})
        except Exception as e:
            log.error(f"Logging Error: {e}")

    try:
        if request.stream:
            async def stream_gen():
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream("POST", f"{OPENROUTER_API_BASE}/chat/completions", json=payload, headers=headers) as resp:
                        if resp.status_code != 200:
                            # Forward high-level error if OR fails early
                            error_text = await resp.aread()
                            log.error(f"Upstream Stream Error {resp.status_code}: {error_text}")
                            yield f"data: {json.dumps({'error': {'message': f'OpenRouter Error {resp.status_code}', 'type': 'upstream_error'}})}\n\n".encode()
                            return

                        gen_id = None
                        async for chunk in resp.aiter_bytes():
                            if not gen_id and chunk.startswith(b"data: "):
                                try:
                                    content = chunk.decode('utf-8').replace('data: ', '').strip()
                                    if content != '[DONE]':
                                        chunk_data = json.loads(content)
                                        gen_id = chunk_data.get("id")
                                except: pass
                            yield chunk
                        
                        finish_latency = (time.time() - start_time) * 1000
                        background_tasks.add_task(log_and_broadcast, gen_id, or_key, 200, finish_latency)

            return StreamingResponse(stream_gen(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient(timeout=120.0) as client:
                try:
                    resp = await client.post(f"{OPENROUTER_API_BASE}/chat/completions", json=payload, headers=headers)
                    resp_json = resp.json()
                    gen_id = resp_json.get("id")
                    finish_latency = (time.time() - start_time) * 1000
                    background_tasks.add_task(log_and_broadcast, gen_id, or_key, resp.status_code, finish_latency)
                    return JSONResponse(content=resp_json, status_code=resp.status_code)
                except Exception as e:
                    log.error(f"Post Error: {e}")
                    finish_latency = (time.time() - start_time) * 1000
                    background_tasks.add_task(log_and_broadcast, None, or_key, 500, finish_latency)
                    raise e
    except Exception as e:
        log.error(f"Forward Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "uptime": int(time.time() - app.state.start_time)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
