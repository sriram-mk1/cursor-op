# Context Optimizer Gateway

Proof-of-concept service that compacts session history and tool outputs before an LLM call using FastAPI + Uvicorn. Everything is in memory (no embeddings or external services), so it is portable for Render or OpenRouter-style deployments.

## Quickstart

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -e .`
3. `uvicorn main:app --reload`

Use the endpoints below to ingest conversation items and request optimized context bundles. The `scripts/benchmark.py` script creates a synthetic fixture, ingests it, and prints latency + estimated token savings.

## API

- `POST /v1/ingest` – body: `{"session_id": "...", "events": [{"role": "...", "source": "...", "content": "..."}]}` – chunks and indexes the payload, classifies roles, deduplicates via SimHash, and maintains a rolling window of chunks per session.
- `POST /v1/optimize` – body: `{"session_id": "...", "query_text": "...", "max_chunks": 12, "target_token_budget": 600}` – runs BM25 retrieval, dedups the top candidates, runs role-aware shrinkers, and returns optimized context stats.
- `GET /v1/stats?session_id=...` – Returns session counts, dedup rate, cache hit rate, and the last optimize snapshot.

## Curl Examples

```bash
curl -X POST localhost:8000/v1/ingest -H "Content-Type: application/json" \
  -d '{"session_id":"alpha","events":[{"role":"user","source":"chat","content":"We need to ship a plan"}]}'

curl -X POST localhost:8000/v1/optimize -H "Content-Type: application/json" \
  -d '{"session_id":"alpha","query_text":"plan for release","max_chunks":8}'

curl localhost:8000/v1/stats?session_id=alpha
```

## Fixtures & Benchmarks

- `python scripts/generate_fixtures.py` writes `fixtures/stress_fixture.json` with long chat, logs, docs, and code snippets for stress. Use those artifacts in your agents to replay heavy history.
- `python scripts/benchmark.py` loads `fixtures/synthetic_fixture.json` (auto-generated) and prints ingestion + optimization latency plus raw vs. optimized token estimates.

## Testing

Run `pytest` for chunking, BM25 retrieval, SimHash dedup, and shrinker coverage.

## Deployment Notes

Keep the service stateless by routing session IDs to the correct worker. Everything is in memory per session, so you can containerize with Uvicorn and scale horizontally on Render/OpenRouter. Tokens are estimated with a regex tokenizer and the optimized bundle includes per-session cache stats for rapid re-use.
