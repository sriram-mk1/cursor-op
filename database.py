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

# Production Environment Variable Loading
load_dotenv() 

class Database:
    def __init__(self):
        url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        logger.info(f"Initializing Database. URL found: {bool(url)}, Key found: {bool(key)}")
        
        if not url or not key:
            logger.error(f"Missing Supabase credentials. URL: {'found' if url else 'MISSING'}, Key: {'found' if key else 'MISSING'}")
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


    def log_request(self, api_key_raw: str, session_id: str, model: str, original_tokens: int, latency_ms: float, reconstruction_log: Dict = None, or_metadata: Dict = None):
        self._check_db()
        hashed = self._hash_key(api_key_raw)
        now = time.time()
        
        # 1. Extract data from OpenRouter metadata or fallback
        if or_metadata:
            t_in = or_metadata.get("tokens_prompt", 0)
            t_out = or_metadata.get("tokens_completion", 0)
            t_cost = or_metadata.get("total_cost", 0)
            t_saved = max(0, original_tokens - t_in)
            
            total_tokens = t_in + t_out
            c_saved = (t_saved * (t_cost / total_tokens)) if total_tokens > 0 else 0
        else:
            t_in, t_out, t_saved, t_cost, c_saved = original_tokens, 0, 0, 0, 0

        try:
            # 2. Check if a conversation row already exists for this session
            existing = self.supabase.table("analytics").select("*").eq("session_id", session_id).execute()
            
            if existing.data:
                # Aggregate! (One Row per Conversation)
                row = existing.data[0]
                analytics_data = {
                    "tokens_in": row["tokens_in"] + t_in,
                    "tokens_out": row["tokens_out"] + t_out,
                    "tokens_saved": row["tokens_saved"] + t_saved,
                    "cost_saved_usd": float(row["cost_saved_usd"] or 0) + c_saved,
                    "total_cost_usd": float(row["total_cost_usd"] or 0) + t_cost,
                    "latency_ms": latency_ms, # Keep latest latency
                    "reconstruction_log": reconstruction_log or row["reconstruction_log"],
                    "timestamp": now,
                    "or_id": or_metadata.get("id") if or_metadata else row["or_id"]
                }
                self.supabase.table("analytics").update(analytics_data).eq("session_id", session_id).execute()
                res_id = row["id"]
            else:
                # Create NEW row for this conversation
                res_id = str(uuid.uuid4())
                analytics_data = {
                    "id": res_id,
                    "hashed_key": hashed,
                    "session_id": session_id,
                    "model": model,
                    "tokens_in": t_in,
                    "tokens_out": t_out,
                    "tokens_saved": t_saved,
                    "latency_ms": latency_ms,
                    "cost_saved_usd": c_saved,
                    "total_cost_usd": t_cost,
                    "reconstruction_log": reconstruction_log or {},
                    "timestamp": now,
                    "or_id": or_metadata.get("id") if or_metadata else None
                }
                self.supabase.table("analytics").insert(analytics_data).execute()

            # 3. Update global API Key aggregate stats
            key_data = self.supabase.table("api_keys").select("total_tokens_saved, total_requests").eq("hashed_key", hashed).single().execute()
            if key_data.data:
                self.supabase.table("api_keys").update({
                    "total_tokens_saved": key_data.data["total_tokens_saved"] + t_saved,
                    "total_requests": key_data.data["total_requests"] + 1
                }).eq("hashed_key", hashed).execute()

        except Exception as e:
            logger.error(f"DB Upsert Error: {e}")
            res_id = str(uuid.uuid4())

        return {
            "id": res_id,
            "session_id": session_id,
            "model": model,
            "tokens_in": t_in,
            "tokens_out": t_out,
            "tokens_saved": t_saved,
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
