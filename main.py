"""
Context Optimizer Gateway
=========================
A simple OpenRouter-compatible proxy that receives AI IDE context,
restructures it, and forwards to OpenRouter.

Phase 1: Logging + Context Manipulation (prove we can intercept and modify)
Phase 2: Add chunking
Phase 3: Add RAG retrieval
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Literal
from dataclasses import dataclass, field, asdict

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import tiktoken


# ============================================================================
# LOGGING SETUP - Clean, colorful, structured
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and better structure."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',      # Reset
        'BOLD': '\033[1m',       # Bold
        'DIM': '\033[2m',        # Dim
    }
    
    def format(self, record):
        # Add timestamp
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        # Get color for level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        dim = self.COLORS['DIM']
        bold = self.COLORS['BOLD']
        
        # Format the message
        level_short = record.levelname[0]  # Just first letter
        
        return f"{dim}{timestamp}{reset} {color}{bold}[{level_short}]{reset} {record.getMessage()}"


def setup_logging():
    """Configure beautiful logging."""
    # Create logger
    logger = logging.getLogger("gateway")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Add colored console handler
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    logger.addHandler(handler)
    
    return logger


log = setup_logging()

# Token limits
MAX_INPUT_TOKENS = 5000  # Hard limit on input tokens


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RequestStats:
    """Track stats for a single request."""
    request_id: str
    timestamp: float = field(default_factory=time.time)
    
    # Input stats
    input_messages: int = 0
    input_tokens: int = 0
    
    # Output stats
    output_messages: int = 0
    output_tokens: int = 0
    
    # Processing
    optimization_applied: bool = False
    processing_time_ms: float = 0
    
    def to_dict(self):
        return asdict(self)


# ============================================================================
# TOKEN COUNTER
# ============================================================================

class TokenCounter:
    """Simple token counter using tiktoken."""
    
    def __init__(self):
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = tiktoken.get_encoding("p50k_base")
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def count_message(self, message: Dict[str, Any]) -> int:
        """Count tokens in a message."""
        tokens = 4  # Message overhead
        content = message.get("content", "")
        
        if isinstance(content, str):
            tokens += self.count(content)
        elif isinstance(content, list):
            # Handle content parts (text, images, etc.)
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    tokens += self.count(part.get("text", ""))
        
        tokens += self.count(message.get("role", ""))
        return tokens
    
    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in message list."""
        total = 3  # Reply priming
        for msg in messages:
            total += self.count_message(msg)
        return total


token_counter = TokenCounter()


# ============================================================================
# CONTEXT MANIPULATOR - The core of what we're testing
# ============================================================================

class ContextManipulator:
    """
    Simple context manipulator.
    For now, just logs what we receive and passes it through.
    Later we'll add: chunking, RAG, etc.
    """
    
    def __init__(self):
        self.request_count = 0
    
    def process(self, messages: List[Dict[str, Any]], stats: RequestStats) -> List[Dict[str, Any]]:
        """
        Process incoming messages with token limit enforcement.
        
        MAX_INPUT_TOKENS = 5000
        If over limit, we truncate messages from the middle,
        keeping: system message, last 2 turns, and the final user query.
        """
        self.request_count += 1
        
        log.info(f"{'='*60}")
        log.info(f"📥 INCOMING REQUEST #{self.request_count}")
        log.info(f"{'='*60}")
        
        # Analyze what we received
        stats.input_messages = len(messages)
        stats.input_tokens = token_counter.count_messages(messages)
        
        log.info(f"📊 Messages: {stats.input_messages}")
        log.info(f"📊 Tokens: {stats.input_tokens:,}")
        
        # Log each message (summarized)
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = self._extract_text(msg.get("content", ""))
            
            # Truncate for logging
            preview = content[:100] + "..." if len(content) > 100 else content
            preview = preview.replace("\n", "\\n")
            
            log.debug(f"  [{i}] {role}: {preview}")
        
        # Identify message types
        system_msgs = [m for m in messages if m.get("role") == "system"]
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        tool_msgs = [m for m in messages if m.get("role") in ("tool", "function")]
        
        log.info(f"📋 Breakdown: system={len(system_msgs)}, user={len(user_msgs)}, "
                f"assistant={len(assistant_msgs)}, tool={len(tool_msgs)}")
        
        # ====================================================================
        # TOKEN LIMIT ENFORCEMENT (5K max)
        # ====================================================================
        if stats.input_tokens > MAX_INPUT_TOKENS:
            log.warning(f"⚠️ Over token limit! {stats.input_tokens:,} > {MAX_INPUT_TOKENS:,}")
            messages = self._truncate_to_limit(messages, MAX_INPUT_TOKENS)
            new_token_count = token_counter.count_messages(messages)
            log.info(f"✂️ Truncated: {stats.input_tokens:,} → {new_token_count:,} tokens")
            stats.optimization_applied = True
        
        # ====================================================================
        # MANIPULATION ZONE
        # For now, just pass through. Later we'll:
        # 1. Chunk the context
        # 2. Do RAG retrieval
        # 3. Reconstruct optimized context
        # ====================================================================
        
        # PROOF OF MANIPULATION: Add a marker to the system message
        # This proves we're actually intercepting and modifying context
        output_messages = []
        marker_added = False
        
        for msg in messages:
            new_msg = dict(msg)  # Copy
            
            # Add marker to system message
            if msg.get("role") == "system" and not marker_added:
                content = self._extract_text(msg.get("content", ""))
                new_msg["content"] = f"{content}\n\n[🔧 Context processed by optimizer | {stats.input_messages} msgs, {stats.input_tokens} tokens]"
                marker_added = True
                log.info("✅ Added optimization marker to system message")
            
            output_messages.append(new_msg)
        
        # If no system message, add one with marker
        if not marker_added:
            output_messages.insert(0, {
                "role": "system",
                "content": f"[🔧 Context processed by optimizer | {stats.input_messages} msgs, {stats.input_tokens} tokens]"
            })
            log.info("✅ Inserted new system message with optimization marker")
        
        # Calculate output stats
        stats.output_messages = len(output_messages)
        stats.output_tokens = token_counter.count_messages(output_messages)
        
        # Did we optimize?
        if stats.output_tokens < stats.input_tokens:
            stats.optimization_applied = True
            saved = stats.input_tokens - stats.output_tokens
            pct = (saved / stats.input_tokens) * 100
            log.info(f"✨ Optimized: {stats.input_tokens:,} → {stats.output_tokens:,} tokens ({pct:.1f}% saved)")
        else:
            log.info(f"📤 Passing through: {stats.output_messages} messages, {stats.output_tokens:,} tokens")
        
        log.info(f"{'='*60}")
        
        return output_messages
    
    def _extract_text(self, content) -> str:
        """Extract text from content (handles string and list formats)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            return " ".join(texts)
        return str(content)
    
    def _truncate_to_limit(self, messages: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """
        Truncate messages to fit within token limit.
        Strategy: Keep system message, last user message, and recent context.
        Remove older middle messages first.
        """
        if not messages:
            return messages
        
        # Separate message types
        system_msg = None
        other_msgs = []
        
        for msg in messages:
            if msg.get("role") == "system" and system_msg is None:
                system_msg = msg
            else:
                other_msgs.append(msg)
        
        # Always keep: system (if exists), last message (the query)
        # Then add from the end until we hit the limit
        result = []
        current_tokens = 0
        
        # Add system message first
        if system_msg:
            sys_tokens = token_counter.count_message(system_msg)
            result.append(system_msg)
            current_tokens += sys_tokens
        
        # Add messages from the end (most recent first)
        messages_to_add = []
        for msg in reversed(other_msgs):
            msg_tokens = token_counter.count_message(msg)
            if current_tokens + msg_tokens <= max_tokens:
                messages_to_add.insert(0, msg)  # Insert at beginning to maintain order
                current_tokens += msg_tokens
            else:
                # Skip this message (too old, over budget)
                log.debug(f"  Dropping message: {msg.get('role')} ({msg_tokens} tokens)")
        
        result.extend(messages_to_add)
        
        log.info(f"📝 Kept {len(result)} of {len(messages)} messages ({current_tokens:,} tokens)")
        return result


context_manipulator = ContextManipulator()


# ============================================================================
# PYDANTIC MODELS (Simplified)
# ============================================================================

class ChatCompletionRequest(BaseModel):
    """OpenRouter-compatible chat completion request."""
    
    model_config = {"extra": "allow"}
    
    # Core
    messages: Optional[List[Any]] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    
    # Generation params
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    
    # OpenRouter specific
    transforms: Optional[List[str]] = None
    route: Optional[str] = None
    
    # Our params
    enable_optimization: Optional[bool] = True


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Context Optimizer Gateway",
    version="2.0.0",
    description="Simple OpenRouter proxy with context manipulation"
)

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def extract_api_key(authorization: str) -> str:
    """Extract API key from Authorization header."""
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
    """
    Main endpoint - receives context from AI IDEs (Kilocode, Cursor, etc.),
    manipulates it, then forwards to OpenRouter.
    """
    start_time = time.time()
    
    # Create request stats
    request_id = f"req_{int(time.time() * 1000)}"
    stats = RequestStats(request_id=request_id)
    
    # Extract API key
    api_key = extract_api_key(authorization)
    
    # Validate
    if not request.messages and not request.prompt:
        raise HTTPException(status_code=400, detail="Either 'messages' or 'prompt' is required")
    
    # Convert prompt to messages if needed
    if request.prompt and not request.messages:
        request.messages = [{"role": "user", "content": request.prompt}]
    
    # Normalize messages to dicts
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
    
    # ========================================================================
    # CONTEXT MANIPULATION
    # ========================================================================
    if request.enable_optimization:
        messages = context_manipulator.process(messages, stats)
    
    # Build OpenRouter request
    request_dict = request.model_dump(exclude_none=True)
    
    # Remove our custom params
    for param in ["enable_optimization"]:
        request_dict.pop(param, None)
    
    # Update with processed messages
    request_dict["messages"] = messages
    request_dict.pop("prompt", None)
    
    # Calculate processing time
    stats.processing_time_ms = (time.time() - start_time) * 1000
    
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": http_referer or "https://context-optimizer.app",
        "X-Title": x_title or "Context Optimizer Gateway"
    }
    
    log.info(f"🚀 Forwarding to OpenRouter (processed in {stats.processing_time_ms:.1f}ms)")
    
    # Stream or regular response
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
                
                if response.status_code != 200:
                    log.error(f"OpenRouter error {response.status_code}: {response.text[:200]}")
                
                # Add our headers
                response_headers = {
                    "X-Context-Request-Id": request_id,
                    "X-Context-Input-Tokens": str(stats.input_tokens),
                    "X-Context-Output-Tokens": str(stats.output_tokens),
                    "X-Context-Processing-Ms": f"{stats.processing_time_ms:.1f}",
                }
                
                return JSONResponse(
                    content=response.json(),
                    status_code=response.status_code,
                    headers=response_headers
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
    """List available models."""
    api_key = extract_api_key(authorization)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{OPENROUTER_API_BASE}/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return JSONResponse(content=response.json(), status_code=response.status_code)


@app.get("/v1/models/{model_id}")
@app.get("/api/v1/models/{model_id}")
async def get_model(model_id: str, authorization: Optional[str] = Header(None)):
    """Get model info."""
    api_key = extract_api_key(authorization)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{OPENROUTER_API_BASE}/models/{model_id}",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return JSONResponse(content=response.json(), status_code=response.status_code)


# ============================================================================
# ANALYTICS ENDPOINTS (OpenRouter Activity API)
# ============================================================================

@app.get("/v1/activity")
@app.get("/api/v1/activity")
@app.get("/activity")
async def get_activity(
    date: Optional[str] = None,
    authorization: Optional[str] = Header(None)
):
    """
    Get accurate usage analytics from OpenRouter.
    This gives us REAL token counts, not our estimates.
    
    Optional: ?date=YYYY-MM-DD to filter by specific date
    """
    api_key = extract_api_key(authorization)
    
    log.info(f"📊 Fetching activity from OpenRouter" + (f" for {date}" if date else ""))
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {}
            if date:
                params["date"] = date
            
            response = await client.get(
                f"{OPENROUTER_API_BASE}/activity",
                params=params,
                headers={"Authorization": f"Bearer {api_key}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                activity = data.get("data", [])
                
                # Log summary
                if activity:
                    total_requests = sum(item.get("requests", 0) for item in activity)
                    total_prompt = sum(item.get("prompt_tokens", 0) for item in activity)
                    total_completion = sum(item.get("completion_tokens", 0) for item in activity)
                    total_cost = sum(item.get("usage", 0) for item in activity)
                    
                    log.info(f"📈 Activity Summary:")
                    log.info(f"   Requests: {total_requests:,}")
                    log.info(f"   Prompt tokens: {total_prompt:,}")
                    log.info(f"   Completion tokens: {total_completion:,}")
                    log.info(f"   Total cost: ${total_cost:.4f}")
            
            return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.HTTPError as e:
        log.error(f"Failed to fetch activity: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch activity: {str(e)}")


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "phase": "1 - Logging & Context Manipulation",
        "requests_processed": context_manipulator.request_count
    }


@app.get("/")
async def root():
    """API info."""
    return {
        "service": "Context Optimizer Gateway",
        "version": "2.0.0",
        "phase": "1 - Logging & Context Manipulation",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        },
        "roadmap": {
            "phase_1": "✅ Logging + Context Manipulation (current)",
            "phase_2": "⏳ Chunking",
            "phase_3": "⏳ RAG Retrieval"
        }
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    
    log.info("=" * 60)
    log.info("🚀 Context Optimizer Gateway v2.0.0")
    log.info("=" * 60)
    log.info(f"📍 Running on port {port}")
    log.info(f"📋 Phase 1: Logging & Context Manipulation")
    log.info("=" * 60)
    
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
