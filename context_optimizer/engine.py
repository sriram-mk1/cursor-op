import os
import json
import logging
import time
import httpx
from typing import List, Dict, Any, Optional, Tuple, Set
import tiktoken

log = logging.getLogger("gateway")
ENCODER = tiktoken.get_encoding("cl100k_base")

class ContextOptimizer:
    """V6-Remote: Lightweight Client for Modal-based Context Optimization."""
    def __init__(self):
        # The URL for the Modal web endpoint
        self.modal_url = os.getenv("MODAL_OPTIMIZER_URL")
        if not self.modal_url:
            log.warning("MODAL_OPTIMIZER_URL not set. Context optimization will be disabled.")
        
        from database import Database
        self.db = Database()

    def optimize(self, user_key: str, session_id: str, current_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        start_time = time.time()
        
        # Fast Path: Scrappy local check for very simple cases
        if not current_messages or len(current_messages) <= 1:
            return current_messages, {"mode": "passthrough"}

        if not self.modal_url:
            return current_messages, {"mode": "no_modal_url"}

        try:
            # Offload heavy ML compute to Modal
            # This keeps Railway RAM usage flat and saves cost via scale-to-zero
            payload = {
                "user_key": user_key,
                "session_id": session_id,
                "current_messages": current_messages
            }
            
            with httpx.Client(timeout=10.0) as client:
                response = client.post(self.modal_url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    # Expecting [optimized_messages, log_metadata]
                    return data[0], data[1]
                else:
                    log.error(f"Modal Error ({response.status_code}): {response.text}")
        except Exception as e:
            log.error(f"Remote Optimization Failed: {e}")

        # Fallback: Return original messages if remote fails
        return current_messages, {
            "mode": "fallback",
            "error": "Remote engine unavailable",
            "overhead_ms": (time.time() - start_time) * 1000
        }
