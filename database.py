import os
import time
import uuid
import hashlib
import logging
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database")

# Try loading from multiple possible locations
load_dotenv() # Default .env
load_dotenv("frontend/.env") # Also try frontend/.env

class Database:
    def __init__(self):
        url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not url or not key:
            logger.error(f"Missing Supabase credentials. URL: {'found' if url else 'MISSING'}, Key: {'found' if key else 'MISSING'}")
            # Don't raise ValueError immediately, allow the app to start but log the error
            # This helps debug if it's an env loading issue
            self.supabase = None
        else:
            try:
                self.supabase: Client = create_client(url, key)
                logger.info("Successfully connected to Supabase")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                self.supabase = None

    def _check_db(self):
        if not self.supabase:
            raise ValueError("Database not initialized. Check Supabase credentials in .env")

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def validate_key(self, key: str) -> Optional[Dict[str, Any]]:
        self._check_db()
        hashed = self._hash_key(key)
        response = self.supabase.table("api_keys").select("*").eq("hashed_key", hashed).execute()
        if response.data:
            return response.data[0]
        return None

    def log_request(self, api_key_raw: str, session_id: str, model: str, tokens_in: int, tokens_out: int, tokens_saved: int, latency_ms: float):
        self._check_db()
        hashed = self._hash_key(api_key_raw)
        now = time.time()
        
        analytics_data = {
            "id": str(uuid.uuid4()),
            "hashed_key": hashed,
            "session_id": session_id,
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_saved": tokens_saved,
            "latency_ms": latency_ms,
            "timestamp": now
        }
        self.supabase.table("analytics").insert(analytics_data).execute()
        
        key_data = self.supabase.table("api_keys").select("total_tokens_saved, total_requests").eq("hashed_key", hashed).single().execute()
        if key_data.data:
            new_saved = key_data.data["total_tokens_saved"] + tokens_saved
            new_reqs = key_data.data["total_requests"] + 1
            self.supabase.table("api_keys").update({
                "total_tokens_saved": new_saved,
                "total_requests": new_reqs
            }).eq("hashed_key", hashed).execute()

        return {
            "id": analytics_data["id"],
            "session_id": session_id,
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_saved": tokens_saved,
            "latency_ms": latency_ms,
            "timestamp": now
        }

    def get_stats(self, api_key_raw: str):
        self._check_db()
        hashed = self._hash_key(api_key_raw)
        key_response = self.supabase.table("api_keys").select("total_tokens_saved, total_requests").eq("hashed_key", hashed).single().execute()
        if not key_response.data:
            return None
        recent_response = self.supabase.table("analytics").select("*").eq("hashed_key", hashed).order("timestamp", desc=True).limit(50).execute()
        return {
            "total_tokens_saved": key_response.data["total_tokens_saved"],
            "total_requests": key_response.data["total_requests"],
            "recent_requests": recent_response.data
        }

    def create_key(self, name: str, user_id: str, openrouter_key: str = "") -> Dict[str, Any]:
        self._check_db()
        raw_key = f"v1-{uuid.uuid4().hex}{uuid.uuid4().hex}"[:48]
        hashed = self._hash_key(raw_key)
        now = time.time()
        
        data = {
            "hashed_key": hashed,
            "raw_key": raw_key,
            "name": name,
            "user_id": user_id,
            "openrouter_key": openrouter_key,
            "created_at": now,
            "total_tokens_saved": 0,
            "total_requests": 0
        }
        self.supabase.table("api_keys").insert(data).execute()
        return {
            "key": raw_key,
            "name": name,
            "created_at": now
        }

    def delete_key(self, hashed_key: str, user_id: str):
        self._check_db()
        self.supabase.table("api_keys").delete().eq("hashed_key", hashed_key).eq("user_id", user_id).execute()

    def list_keys(self, user_id: str) -> List[Dict[str, Any]]:
        self._check_db()
        response = self.supabase.table("api_keys").select("name, created_at, total_tokens_saved, total_requests, raw_key, hashed_key").eq("user_id", user_id).order("created_at", desc=True).execute()
        return response.data

    def update_provider_key(self, v1_key_or_hash: str, user_id: str, openrouter_key: str):
        self._check_db()
        if v1_key_or_hash.startswith("v1-"):
            hashed = self._hash_key(v1_key_or_hash)
        else:
            hashed = v1_key_or_hash
        self.supabase.table("api_keys").update({"openrouter_key": openrouter_key}).eq("hashed_key", hashed).eq("user_id", user_id).execute()
