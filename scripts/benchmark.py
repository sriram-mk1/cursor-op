import json
import random
import textwrap
import time
from pathlib import Path

from context_optimizer import ContextOptimizer


def _build_synthetic_events() -> list[dict]:
    user_chatter = [
        "User: We need a plan to scale the ingestion pipeline before the next release.",
        "Assistant: I'll audit the tools and surface blockers as a report.",
        "User: What were the previous diagnostics on the payment queue?",
        "Assistant: The earlier logs showed intermittent disk pressure.",
    ]
    events = []
    for idx, line in enumerate(user_chatter):
        role = "user" if idx % 2 == 0 else "assistant"
        events.append({"role": role, "source": "chat", "content": line})

    for log_index in range(5):
        log = "ERROR" if log_index % 2 else "WARN"
        content = "\n".join(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ')} {log} disk pressure event at node {log_index}"
            for _ in range(3)
        )
        events.append({"role": "tool", "source": "system.log", "content": content})

    doc = textwrap.dedent(
        """
        Background: The payments service thrives on a queue that must stay below 30s latency.
        Strategy: Focus on snapshot isolation and avoid repeated calibration between retries.
        """
    ).strip()
    events.append({"role": "tool", "source": "documentations", "content": doc})

    code = textwrap.dedent(
        """
        def process(event):
            if event.retry_count > MAX:
                logger.error("Retry exceeded")
                return False
            # TODO: handle bulk insert
            return scheduler.enqueue(event)
        """
    ).strip()
    events.append({"role": "tool", "source": "code_snippet", "content": code})

    # Add some random chat turns
    for i in range(8):
        events.append(
            {
                "role": "user" if i % 2 else "assistant",
                "source": f"chat.turns.{i}",
                "content": f"Random turn {i} with note {random.randint(1, 100)}",
            }
        )
    return events


def generate_fixture(path: Path) -> dict:
    session_id = "benchmark-session"
    events = _build_synthetic_events()
    payload = {"session_id": session_id, "events": events}
    path.write_text(json.dumps(payload, indent=2))
    return payload


def load_fixture(path: Path) -> dict:
    if not path.exists():
        return generate_fixture(path)
    return json.loads(path.read_text())


def run():
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "synthetic_fixture.json"
    fixture = load_fixture(fixture_path)
    optimizer = ContextOptimizer()
    ingest_start = time.perf_counter()
    optimizer.ingest(fixture["session_id"], fixture["events"])
    ingest_latency = (time.perf_counter() - ingest_start) * 1000

    query = "payment queue disk pressure plan"
    optimize_start = time.perf_counter()
    optimized = optimizer.optimize(fixture["session_id"], query, max_chunks=8, cache_ttl_sec=30)
    optimize_latency = (time.perf_counter() - optimize_start) * 1000

    raw_tokens = optimized["raw_token_est"]
    optimized_tokens = optimized["optimized_token_est"]
    savings = raw_tokens - optimized_tokens

    print("Benchmark results:")
    print(f"  Ingest latency: {ingest_latency:.2f} ms")
    print(f"  Optimize latency: {optimize_latency:.2f} ms")
    print(f"  Raw tokens: {raw_tokens}")
    print(f"  Optimized tokens: {optimized_tokens}")
    print(f"  Estimated savings: {savings} tokens")


if __name__ == "__main__":
    run()
