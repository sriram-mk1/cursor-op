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

    def get_or_create_conversation(self, session_id: str, v1_key: str, user_id: str = None) -> str:
        self._check_db()
        hashed = self._hash_key(v1_key)
        # Check if conversation exists for this session
        res = self.supabase.table("conversations").select("id").eq("session_id", session_id).execute()
        if res.data:
            convo_id = res.data[0]["id"]
            self.supabase.table("conversations").update({"last_request_at": time.time()}).eq("id", convo_id).execute()
            return convo_id
        
        # Create new conversation
        convo_id = str(uuid.uuid4())
        self.supabase.table("conversations").insert({
            "id": convo_id,
            "session_id": session_id,
            "hashed_key": hashed,
            "user_id": user_id,
            "title": f"Conversation {session_id[:8]}",
            "last_request_at": time.time()
        }).execute()
        return convo_id


    def log_request(self, api_key_raw: str, session_id: str, model: str, original_tokens: int, latency_ms: float, reconstruction_log: Dict = None, or_metadata: Dict = None, raw_messages: List = None, response_message: Dict = None):
        self._check_db()
        hashed = self._hash_key(api_key_raw)
        now = time.time()
        
        # 1. Extract data from OpenRouter metadata or fallback
        if or_metadata:
            t_in = or_metadata.get("tokens_prompt", 0)
            t_out = or_metadata.get("tokens_completion", 0)
            t_cost = or_metadata.get("total_cost", 0)
            # V4: Potential total tokens (unoptimized) vs actual tokens used
            t_saved = max(0, original_tokens - t_in)
            
            total_tokens = t_in + t_out
            c_saved = (t_saved * (t_cost / total_tokens)) if total_tokens > 0 else 0
        else:
            t_in, t_out, t_saved, t_cost, c_saved = original_tokens, 0, 0, 0, 0

        try:
            # 0. Get the Conversation ID
            convo_id = self.get_or_create_conversation(session_id, api_key_raw)

            analytics_data = {
                "hashed_key": hashed,
                "conversation_id": convo_id,
                "session_id": session_id,
                "model": model,
                "tokens_in": t_in,
                "tokens_out": t_out,
                "tokens_saved": t_saved,
                "latency_ms": latency_ms,
                "cost_saved_usd": float(c_saved),
                "total_cost_usd": float(t_cost),
                "reconstruction_log": reconstruction_log or {},
                "timestamp": now,
                "or_id": or_metadata.get("id") if or_metadata else None,
                "raw_messages": raw_messages, # Full observability
                "response_message": response_message # Full observability
            }
            # 2. Create NEW row (Observability: One row per request)
            res_id = str(uuid.uuid4())
            analytics_data["id"] = res_id
            self.supabase.table("analytics").insert(analytics_data).execute()

            # 3. Update global API Key aggregate stats
            try:
                key_data = self.supabase.table("api_keys").select("total_tokens_saved, total_requests").eq("hashed_key", hashed).single().execute()
                if key_data.data:
                    self.supabase.table("api_keys").update({
                        "total_tokens_saved": (key_data.data.get("total_tokens_saved") or 0) + t_saved,
                        "total_requests": (key_data.data.get("total_requests") or 0) + 1
                    }).eq("hashed_key", hashed).execute()
            except Exception as ke:
                logger.warning(f"Could not update key stats: {ke}")

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
            "timestamp": now,
            "cost_saved_usd": c_saved,
            "total_cost_usd": t_cost,
            "potential_total": original_tokens
        }

    def get_stats(self, api_key_raw: str):
        self._check_db()
        hashed = self._hash_key(api_key_raw)
        try:
            key_response = self.supabase.table("api_keys").select("total_tokens_saved, total_requests").eq("hashed_key", hashed).single().execute()
            if not key_response.data:
                return None
            
            recent_response = self.supabase.table("analytics").select("*").eq("hashed_key", hashed).order("timestamp", desc=True).limit(50).execute()
            return {
                "total_tokens_saved": key_response.data.get("total_tokens_saved") or 0,
                "total_requests": key_response.data.get("total_requests") or 0,
                "recent_requests": recent_response.data or []
            }
        except Exception:
            return None

    def get_user_stats(self, user_id: str):
        """Aggregate stats + fetch recent conversations."""
        self._check_db()
        keys = self.supabase.table("api_keys").select("hashed_key, total_tokens_saved, total_requests").eq("user_id", user_id).execute()
        if not keys.data:
            return {"total_tokens_saved": 0, "total_requests": 0, "recent_requests": [], "recent_conversations": []}
        
        hashes = [k["hashed_key"] for k in keys.data]
        total_saved = sum(k["total_tokens_saved"] for k in keys.data)
        total_reqs = sum(k["total_requests"] for k in keys.data)
        
        # 1. Latest Analytics
        recent = self.supabase.table("analytics").select("*").in_("hashed_key", hashes).order("timestamp", desc=True).limit(25).execute()
        
        # 2. Latest Conversations
        convos = self.supabase.table("conversations").select("*").in_("hashed_key", hashes).order("last_request_at", desc=True).limit(10).execute()
        
        return {
            "total_tokens_saved": total_saved,
            "total_requests": total_reqs,
            "recent_requests": recent.data,
            "recent_conversations": convos.data
        }

    def get_conversation_detail(self, convo_id: str):
        self._check_db()
        convo = self.supabase.table("conversations").select("*").eq("id", convo_id).single().execute()
        requests = self.supabase.table("analytics").select("*").eq("conversation_id", convo_id).order("timestamp", asc=True).execute()
        return {
            "conversation": convo.data,
            "requests": requests.data
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

    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        self._check_db()
        try:
            response = self.supabase.table("sessions").select("data").eq("session_id", session_id).single().execute()
            if response.data:
                return response.data["data"]
        except Exception as e:
            # logger.warning(f"Session lookup failed for {session_id}: {e}")
            pass
        return None

    def save_session_state(self, session_id: str, data: Dict[str, Any]):
        self._check_db()
        try:
            now = time.time()
            # Use upsert provided by supabase-py (uses PostgreSQL ON CONFLICT)
            # Requires a primary key on the table (session_id)
            self.supabase.table("sessions").upsert({
                "session_id": session_id,
                "data": data,
                "updated_at": now
            }, on_conflict="session_id").execute()
        except Exception as e:
            logger.error(f"Session save failed for {session_id}: {e}")
            raise e
