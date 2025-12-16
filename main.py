from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from context_optimizer import ContextOptimizer


class EventSchema(BaseModel):
    role: str
    source: str
    content: str
    ts: Optional[float]


class IngestSchema(BaseModel):
    session_id: str
    events: List[EventSchema]


class OptimizeSchema(BaseModel):
    session_id: str
    query_text: str
    max_chunks: Optional[int] = 12
    target_token_budget: Optional[int]
    cache_ttl_sec: Optional[int] = 60


app = FastAPI(title="Context Optimizer Gateway", version="0.1.0")
optimizer = ContextOptimizer()


@app.post("/v1/ingest")
async def ingest(payload: IngestSchema) -> dict:
    details = optimizer.ingest(payload.session_id, [event.dict() for event in payload.events])
    return {"status": "ok", "details": details}


@app.post("/v1/optimize")
async def optimize(payload: OptimizeSchema) -> dict:
    result = optimizer.optimize(
        payload.session_id,
        payload.query_text,
        max_chunks=payload.max_chunks or 12,
        target_token_budget=payload.target_token_budget,
        cache_ttl_sec=payload.cache_ttl_sec or 60,
    )
    return result


@app.get("/v1/stats")
async def stats(session_id: str) -> dict:
    return optimizer.get_stats(session_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
