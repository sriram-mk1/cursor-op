import os
import time
import logging
from typing import Any, Dict, List, Optional, Union, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from context_optimizer import ContextOptimizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("context-optimizer-gateway")


# ===== OpenRouter-Compatible Request Models =====

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageContentPart(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: Dict[str, str]  # {url: str, detail?: str}


ContentPart = Union[TextContent, ImageContentPart]


class Message(BaseModel):
    model_config = {"extra": "allow"}
    
    role: str
    content: Union[str, List[ContentPart]]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None  # For tool role messages


class FunctionDescription(BaseModel):
    description: Optional[str] = None
    name: str
    parameters: Dict[str, Any]  # JSON Schema object


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDescription


class ToolChoiceFunction(BaseModel):
    type: Literal["function"] = "function"
    function: Dict[str, str]  # {name: str}


ToolChoice = Union[Literal["none", "auto"], ToolChoiceFunction]


class ResponseFormat(BaseModel):
    type: Literal["json_object", "json_schema"] = "json_object"
    json_schema: Optional[Dict[str, Any]] = None


class ProviderPreferences(BaseModel):
    model_config = {"extra": "allow"}
    # Provider routing options - pass through to OpenRouter
    allow_fallbacks: Optional[bool] = None
    require_parameters: Optional[bool] = None
    data_collection: Optional[str] = None
    order: Optional[List[str]] = None


class PredictionContent(BaseModel):
    type: Literal["content"] = "content"
    content: str


class DebugOptions(BaseModel):
    echo_upstream_body: Optional[bool] = None


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow", "arbitrary_types_allowed": True}
    
    # Core parameters (either messages or prompt required)
    messages: Optional[List[Any]] = None  # Accept any message format
    prompt: Optional[str] = None
    model: Optional[str] = None  # If omitted, uses user's default
    
    # Response formatting
    response_format: Optional[Any] = None  # Accept any format
    
    # Stop sequences
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    
    # Token limits
    max_tokens: Optional[int] = None
    
    # Temperature and sampling
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    top_a: Optional[float] = None
    min_p: Optional[float] = None
    
    # Penalties
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    
    # Tool calling
    tools: Optional[List[Any]] = None  # Accept any tool format
    tool_choice: Optional[Any] = None  # Accept any choice format
    parallel_tool_calls: Optional[bool] = None
    
    # Advanced parameters
    seed: Optional[int] = None
    logit_bias: Optional[Dict] = None  # Accept any dict
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    
    # Predicted outputs (OpenAI latency optimization)
    prediction: Optional[Any] = None  # Accept any prediction format
    
    # OpenRouter-specific parameters
    transforms: Optional[List[str]] = None
    models: Optional[List[str]] = None  # For fallback routing
    route: Optional[str] = None  # Accept any route string
    provider: Optional[Any] = None  # Accept any provider format
    user: Optional[str] = None
    
    # Debug options (streaming only)
    debug: Optional[Any] = None  # Accept any debug format
    
    # Context Optimizer parameters (custom, won't be sent to OpenRouter)
    enable_optimization: Optional[bool] = True
    target_token_budget: Optional[int] = None
    max_chunks: Optional[int] = 12


app = FastAPI(
    title="Context Optimizer Gateway",
    version="1.0.0",
    description="OpenRouter-compatible gateway with intelligent context optimization"
)
optimizer = ContextOptimizer()

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def extract_api_key(authorization: str):
    """Extract OpenRouter API key from Authorization header"""
    if not authorization:
        raise HTTPException(
            status_code=401, 
            detail="Missing Authorization header"
        )
    
    # Clean and parse authorization header
    authorization = authorization.strip()
    
    # Handle both "Bearer token" and just "token" formats
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()  # Skip "Bearer "
    else:
        # Some clients might send token directly
        token = authorization.strip()
    
    if not token:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    return token


def extract_text_content(content: Union[str, List[ContentPart]]) -> str:
    """Extract text from message content (handles both string and ContentPart array)"""
    if isinstance(content, str):
        return content
    
    # Extract text from ContentPart array
    text_parts = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        elif hasattr(part, "type") and part.type == "text":
            text_parts.append(part.text)
    
    return " ".join(text_parts)


@app.post("/v1/chat/completions")
@app.post("/api/v1/chat/completions")  # Also support OpenRouter's path
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None, alias="authorization"),
    http_referer: Optional[str] = Header(None, alias="http-referer"),
    x_title: Optional[str] = Header(None, alias="x-title"),
):
    """
    OpenRouter-compatible chat completions with intelligent context optimization.
    
    Supports all OpenRouter parameters plus context optimization:
    - enable_optimization: Enable/disable optimization (default: true)
    - target_token_budget: Max tokens for optimized context
    - max_chunks: Max context chunks to retain (default: 12)
    """
    # Extract OpenRouter API key
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    openrouter_api_key = extract_api_key(authorization)
    
    # Validate request
    if not request.messages and not request.prompt:
        raise HTTPException(
            status_code=400,
            detail="Either 'messages' or 'prompt' is required"
        )
    
    # Convert to messages format if prompt was provided
    if request.prompt and not request.messages:
        request.messages = [{"role": "user", "content": request.prompt}]
    
    # Extract messages - handle any format
    messages = []
    for msg in request.messages:
        if isinstance(msg, dict):
            messages.append(msg)
        else:
            # Handle Pydantic models or other objects
            try:
                if hasattr(msg, 'model_dump'):
                    messages.append(msg.model_dump(exclude_none=True))
                elif hasattr(msg, 'dict'):
                    messages.append(msg.dict(exclude_none=True))
                else:
                    messages.append(dict(msg))
            except:
                # Fallback: treat as dict
                messages.append(msg if isinstance(msg, dict) else {"role": "user", "content": str(msg)})
    
    # Track original state
    original_message_count = len(messages)
    optimization_applied = False
    optimization_stats = {}
    
    # Apply context optimization if enabled and worthwhile
    if request.enable_optimization and len(messages) > 3:
        try:
            # Generate stable session ID
            session_id = request.user or f"session_{hash(str(messages[0]))}"
            
            # Prepare events for ingestion (history except last message)
            events = []
            for msg in messages[:-1]:
                content = extract_text_content(msg.get("content", ""))
                if content:
                    events.append({
                        "role": msg.get("role", "user"),
                        "source": "chat",
                        "content": content,
                        "ts": time.time()
                    })
            
            # Ingest conversation history
            if events:
                optimizer.ingest(session_id, events)
            
            # Get latest message content as query
            latest_msg = messages[-1]
            query_text = extract_text_content(latest_msg.get("content", ""))
            
            # Optimize context
            optimization_result = optimizer.optimize(
                session_id,
                query_text,
                max_chunks=request.max_chunks or 12,
                target_token_budget=request.target_token_budget,
                cache_ttl_sec=300
            )
            
            # Apply optimization if we got useful results
            if optimization_result.get("optimized_context"):
                optimization_applied = True
                optimization_stats = {
                    "raw_tokens": optimization_result.get("raw_token_est", 0),
                    "optimized_tokens": optimization_result.get("optimized_token_est", 0),
                    "percent_saved": optimization_result.get("percent_saved_est", 0)
                }
                
                # Build optimized context string
                optimized_chunks = optimization_result.get("optimized_context", [])
                context_parts = []
                for chunk in optimized_chunks:
                    summary = chunk.get("summary", "")
                    if summary:
                        context_parts.append(summary)
                
                optimized_context = "\n\n".join(context_parts)
                
                # Find system message position
                system_msg_idx = None
                for idx, msg in enumerate(messages):
                    if msg.get("role") == "system":
                        system_msg_idx = idx
                        break
                
                # Inject optimized context
                if system_msg_idx is not None:
                    # Append to existing system message
                    existing_content = extract_text_content(messages[system_msg_idx].get("content", ""))
                    messages[system_msg_idx]["content"] = (
                        f"{existing_content}\n\n"
                        f"[Previous conversation context - optimized]:\n{optimized_context}"
                    )
                else:
                    # Create new system message
                    messages.insert(0, {
                        "role": "system",
                        "content": f"[Previous conversation context - optimized]:\n{optimized_context}"
                    })
                
                # Keep only the last user message after system context
                messages = [messages[0], messages[-1]] if len(messages) > 1 else messages
                
                logger.info(
                    f"Optimization: {original_message_count} msgs → {len(messages)} msgs | "
                    f"Tokens: {optimization_stats['raw_tokens']} → {optimization_stats['optimized_tokens']} "
                    f"({optimization_stats['percent_saved']:.1f}% saved)"
                )
        
        except Exception as e:
            # Don't fail request if optimization fails, just log and continue
            logger.warning(f"Context optimization failed: {e}")
            optimization_applied = False
    
    # Build OpenRouter request (exclude our custom params)
    request_dict = request.model_dump(exclude_none=True)
    optimization_params = {"enable_optimization", "target_token_budget", "max_chunks"}
    openrouter_request = {k: v for k, v in request_dict.items() if k not in optimization_params}
    
    # Update with optimized messages
    openrouter_request["messages"] = messages
    
    # Remove prompt if we converted it to messages
    openrouter_request.pop("prompt", None)
    
    # Prepare headers for OpenRouter
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": http_referer or "https://cursor-op.onrender.com",
        "X-Title": x_title or "Context Optimizer Gateway"
    }
    
    # Debug log (hide most of key for security)
    key_preview = f"{openrouter_api_key[:10]}...{openrouter_api_key[-4:]}" if len(openrouter_api_key) > 14 else "***"
    logger.debug(f"Forwarding to OpenRouter with key: {key_preview}")
    
    # Call OpenRouter
    if request.stream:
        # Streaming response - keep client alive during streaming
        async def generate():
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{OPENROUTER_API_BASE}/chat/completions",
                        json=openrouter_request,
                        headers=headers
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                # Yield error in SSE format
                error_data = f'data: {{"error": "{str(e)}"}}\n\n'
                yield error_data.encode()
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{OPENROUTER_API_BASE}/chat/completions",
                    json=openrouter_request,
                    headers=headers
                )
                
                # Log errors from OpenRouter
                if response.status_code != 200:
                    error_body = response.text
                    logger.error(f"OpenRouter error {response.status_code}: {error_body}")
                
                # Build response headers
                response_headers = {}
                if optimization_applied:
                    response_headers["X-Context-Optimization"] = "enabled"
                    response_headers["X-Context-Original-Messages"] = str(original_message_count)
                    response_headers["X-Context-Optimized-Messages"] = str(len(messages))
                    response_headers["X-Context-Token-Savings"] = f"{optimization_stats.get('percent_saved', 0):.1f}%"
                
                # Return exact OpenRouter response (including errors)
                return JSONResponse(
                    content=response.json(),
                    status_code=response.status_code,
                    headers=response_headers
                )
        
        except httpx.HTTPError as e:
            logger.error(f"OpenRouter request failed: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to reach OpenRouter: {str(e)}"
            )


@app.get("/v1/models")
@app.get("/api/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    """List available OpenRouter models"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    openrouter_api_key = extract_api_key(authorization)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{OPENROUTER_API_BASE}/models",
                headers={"Authorization": f"Bearer {openrouter_api_key}"}
            )
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch models: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch models from OpenRouter")


@app.get("/v1/models/{model_id}")
@app.get("/api/v1/models/{model_id}")
async def get_model(model_id: str, authorization: Optional[str] = Header(None)):
    """Get specific model info (proxy to OpenRouter)"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    openrouter_api_key = extract_api_key(authorization)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{OPENROUTER_API_BASE}/models/{model_id}",
                headers={"Authorization": f"Bearer {openrouter_api_key}"}
            )
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch model {model_id}: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch model from OpenRouter")


@app.get("/v1/models/{model_id}/endpoints")
@app.get("/api/v1/models/{model_id}/endpoints")
async def get_model_endpoints(model_id: str, authorization: Optional[str] = Header(None)):
    """Get model endpoints (proxy to OpenRouter)"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    openrouter_api_key = extract_api_key(authorization)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{OPENROUTER_API_BASE}/models/{model_id}/endpoints",
                headers={"Authorization": f"Bearer {openrouter_api_key}"}
            )
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch endpoints for {model_id}: {e}")
        # Return empty endpoints if not supported
        return JSONResponse(
            content={"endpoints": []},
            status_code=200
        )


@app.get("/api/v1/generation")
async def get_generation(
    id: str,
    authorization: Optional[str] = Header(None)
):
    """Query generation stats by ID (OpenRouter passthrough)"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    openrouter_api_key = extract_api_key(authorization)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{OPENROUTER_API_BASE}/generation",
                params={"id": id},
                headers={"Authorization": f"Bearer {openrouter_api_key}"}
            )
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch generation: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch generation from OpenRouter")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "context-optimizer-gateway",
        "version": "1.0.0",
        "features": [
            "BM25 retrieval",
            "SimHash deduplication",
            "Role-aware shrinking",
            "Token estimation",
            "Intelligent caching"
        ]
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Context Optimizer Gateway",
        "version": "1.0.0",
        "description": "OpenRouter-compatible API gateway with intelligent context optimization",
        "optimization_techniques": [
            "BM25 retrieval for relevance scoring",
            "SimHash deduplication for content similarity",
            "Role-aware context shrinking",
            "Token budget enforcement",
            "Query-based caching"
        ],
        "api_compatibility": "Full OpenRouter API support",
        "endpoints": {
            "chat_completions": "/v1/chat/completions or /api/v1/chat/completions",
            "models": "/v1/models or /api/v1/models",
            "generation": "/api/v1/generation",
            "health": "/health"
        },
        "setup": {
            "base_url": "https://cursor-op.onrender.com",
            "api_key": "Your OpenRouter API key",
            "compatibility": "Works with Cursor, VS Code, Continue, and any OpenAI-compatible client"
        },
        "optimization": {
            "enabled_by_default": True,
            "triggers": "Automatic for 4+ messages",
            "disable": "Set enable_optimization: false in request",
            "parameters": {
                "enable_optimization": "Enable/disable optimization (default: true)",
                "target_token_budget": "Max tokens for optimized context",
                "max_chunks": "Max context chunks to retain (default: 12)"
            }
        },
        "openrouter_features": [
            "All parameters supported (temperature, top_p, top_k, etc.)",
            "Tool calling",
            "Response formatting (JSON mode)",
            "Assistant prefill",
            "Streaming",
            "Model routing & fallbacks",
            "Provider preferences"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting Context Optimizer Gateway v1.0.0 on port {port}")
    logger.info("Features: BM25 retrieval | SimHash dedup | Role-aware shrinking")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
