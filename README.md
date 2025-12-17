# Context Optimizer (Lite)

A lightweight, high-performance RAG (Retrieval Augmented Generation) system for optimizing chat context.

## Features

- **In-Memory RAG**: Uses BM25 (Best Matching 25) algorithm for state-of-the-art text retrieval without heavy vector databases.
- **Zero Heavy Dependencies**: No PyTorch, Transformers, or ChromaDB. Runs on pure Python + NumPy.
- **Blazing Fast**: Context optimization takes < 1ms.
- **Token Aware**: Uses `tiktoken` to strictly enforce token budgets.
- **Session Based**: Manages conversation history in-memory for the duration of the session.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running the Server

```bash
python main.py
```

The server exposes an OpenRouter-compatible API at `/v1/chat/completions`.

### Running the Test

```bash
python test_rag_system.py
```

This will generate a mock conversation, ingest it, and demonstrate the context shrinking capabilities with performance stats.

## Architecture

1.  **Ingest**: Conversation history is tokenized and stored in memory.
2.  **Optimize**:
    *   Incoming query is tokenized.
    *   BM25 scores all history chunks against the query.
    *   Top K relevant chunks are selected.
    *   Chunks are filtered to fit the `target_token_budget`.
    *   Selected chunks are re-ordered chronologically to maintain narrative flow.
3.  **Inject**: The optimized context is injected into the system prompt.
