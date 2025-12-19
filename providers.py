import json
import logging
import os
from typing import List, Dict, Any, AsyncGenerator, Optional
import httpx
from fastapi import HTTPException

log = logging.getLogger("gateway")

# --- Provider Constants ---
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

class ProviderHandler:
    @staticmethod
    def determine_provider(api_key: str, model: str) -> str:
        """
        Determine the provider based on the API key format or model name.
        """
        if api_key.startswith("AIza"):
            return "gemini"
        elif api_key.startswith("sk-ant"):
            return "anthropic"
        elif api_key.startswith("sk-or"):
            return "openrouter"
        
        # Fallback based on model name if key is ambiguous or standard 'sk-'
        if "gemini" in model.lower():
            return "gemini"
        elif "claude" in model.lower():
            return "anthropic"
            
        return "openrouter"

    @staticmethod
    async def stream_gemini(
        api_key: str, 
        model: str, 
        messages: List[Dict[str, Any]], 
        temperature: float = 0.7
    ) -> AsyncGenerator[bytes, None]:
        url = f"{GEMINI_BASE_URL}/{model}:streamGenerateContent?key={api_key}"
        
        # Translate messages to Gemini format
        contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
                continue
                
            # Map roles: user -> user, assistant -> model
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature
            }
        }
        
        if system_instruction:
            payload["system_instruction"] = system_instruction

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=payload, headers={"Content-Type": "application/json"}) as resp:
                if resp.status_code != 200:
                    error_text = await resp.read()
                    log.error(f"Gemini Error {resp.status_code}: {error_text}")
                    yield json.dumps({"error": f"Gemini Error: {error_text.decode()}"}).encode()
                    return

                async for chunk in resp.aiter_lines():
                    if not chunk.strip(): continue
                    # Gemini SSE format is slightly different, usually returns a JSON object per line
                    # But the streamGenerateContent returns a list of objects in a JSON array structure if not careful, 
                    # OR standard SSE if requested? 
                    # The docs say "Uses Server-Sent Events (SSE)".
                    # Actually, the REST API returns a JSON list stream, usually needing parsing.
                    # Let's assume standard SSE 'data: ' prefix handling if it behaves like that, 
                    # but Gemini REST often returns a partial JSON list.
                    # Wait, the docs say: "The response body contains a stream of GenerateContentResponse instances."
                    
                    # For simplicity in this proxy, we might need to normalize the output to OpenAI format 
                    # so the client (frontend) understands it.
                    
                    # Parsing Gemini chunk:
                    try:
                        # Remove "data: " if present
                        clean_chunk = chunk.removeprefix("data: ").strip()
                        if not clean_chunk: continue
                        
                        data = json.loads(clean_chunk)
                        # Extract text
                        text = ""
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                text = parts[0].get("text", "")
                        
                        # Convert to OpenAI chunk format
                        openai_chunk = {
                            "id": "chatcmpl-gemini",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(openai_chunk)}\n\n".encode()
                    except Exception:
                        pass
                
                yield b"data: [DONE]\n\n"

    @staticmethod
    async def send_gemini(
        api_key: str, 
        model: str, 
        messages: List[Dict[str, Any]],
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        url = f"{GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"
        
        contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
                continue
                
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature
            }
        }
        
        if system_instruction:
            payload["system_instruction"] = system_instruction

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"Gemini Error: {resp.text}")
            
            data = resp.json()
            # Map to OpenAI response
            text = ""
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text = parts[0].get("text", "")
            
            return {
                "id": "chatcmpl-gemini",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0, # TODO: Extract if available
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

    @staticmethod
    async def stream_anthropic(
        api_key: str, 
        model: str, 
        messages: List[Dict[str, Any]],
        temperature: float = 0.7
    ) -> AsyncGenerator[bytes, None]:
        url = f"{ANTHROPIC_BASE_URL}/messages"
        
        anthropic_msgs = []
        system_prompt = ""
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "system":
                system_prompt = content
                continue
                
            anthropic_msgs.append({
                "role": role,
                "content": content
            })

        payload = {
            "model": model,
            "messages": anthropic_msgs,
            "max_tokens": 4096,
            "stream": True,
            "temperature": temperature
        }
        
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    error_text = await resp.read()
                    log.error(f"Anthropic Error {resp.status_code}: {error_text}")
                    yield json.dumps({"error": f"Anthropic Error: {error_text.decode()}"}).encode()
                    return

                async for chunk in resp.aiter_lines():
                    if not chunk.strip(): continue
                    if not chunk.startswith("data: "): continue
                    
                    data_str = chunk.removeprefix("data: ").strip()
                    if data_str == "[DONE]": 
                        yield b"data: [DONE]\n\n"
                        break
                        
                    try:
                        data = json.loads(data_str)
                        type_ = data.get("type")
                        
                        text = ""
                        if type_ == "content_block_delta":
                            text = data.get("delta", {}).get("text", "")
                        
                        if text:
                            openai_chunk = {
                                "id": "chatcmpl-anthropic",
                                "object": "chat.completion.chunk",
                                "created": 0,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(openai_chunk)}\n\n".encode()
                    except Exception:
                        pass

    @staticmethod
    async def send_anthropic(
        api_key: str, 
        model: str, 
        messages: List[Dict[str, Any]],
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        url = f"{ANTHROPIC_BASE_URL}/messages"
        
        anthropic_msgs = []
        system_prompt = ""
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "system":
                system_prompt = content
                continue
                
            anthropic_msgs.append({
                "role": role,
                "content": content
            })

        payload = {
            "model": model,
            "messages": anthropic_msgs,
            "max_tokens": 4096,
            "temperature": temperature
        }
        
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"Anthropic Error: {resp.text}")
            
            data = resp.json()
            content = data.get("content", [])
            text = ""
            if content:
                text = content[0].get("text", "")
                
            return {
                "id": data.get("id"),
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
                    "total_tokens": 0
                }
            }
