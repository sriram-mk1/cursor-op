import json
import random
import textwrap
import time
from pathlib import Path


def build_fixture() -> dict:
    session_id = "stress-session"
    events = []
    for idx in range(30):
        role = "user" if idx % 2 else "assistant"
        events.append(
            {
                "role": role,
                "source": f"remixer.chat.{idx}",
                "content": f"Turn {idx} exploring goals {random.choice(['scale', 'stability', 'tickets'])}.",
                "ts": time.time(),
            }
        )
    log_block = "\n".join(
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ')} ERROR - node-{node} - failure {random.randint(1,100)}"
        for node in range(20)
    )
    events.append({"role": "tool", "source": "system.log", "content": log_block})

    doc_text = textwrap.dedent(
        """
        ## Service Design
        The gateway is built to redirect context, shrink conversations, and improve prompt building.
        Key insights include rolling windows, BM25 retrieval, and role-based compaction.
        """
    ).strip()
    events.append({"role": "tool", "source": "guide", "content": doc_text})

    code_block = textwrap.dedent(
        """
        class Gateway:
            def route(self, context):
                if not context:
                    raise ValueError("missing context")
                # aggregator takes inbound data
                return self.transform(context)
        """
    ).strip()
    events.append({"role": "tool", "source": "codebase", "content": code_block})
    return {"session_id": session_id, "events": events}


def main():
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "stress_fixture.json"
    data = build_fixture()
    fixture_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {fixture_path} with {len(data['events'])} events.")


if __name__ == "__main__":
    main()
