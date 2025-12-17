import time
import logging
from context_optimizer import ContextOptimizer

# Setup logging
logging.basicConfig(level=logging.INFO)

def main():
    print("="*50)
    print("TESTING LIGHTWEIGHT RAG SYSTEM (BM25)")
    print("="*50)

    # 1. Initialize
    print("\n[1] Initializing ContextOptimizer...")
    start_init = time.time()
    optimizer = ContextOptimizer()
    print(f"    Initialized in {time.time() - start_init:.4f} seconds.")

    # 2. Mock Data
    print("\n[2] Generating Mock Conversation History...")
    session_id = "test_session_bm25"
    
    # A long conversation about building a web app
    conversation = [
        {"role": "user", "content": "I want to build a Python web app using FastAPI."},
        {"role": "assistant", "content": "FastAPI is a great choice. It's modern, fast (high-performance), and easy to learn. What specific features do you need?"},
        {"role": "user", "content": "I need user authentication, a database connection, and a background task worker."},
        {"role": "assistant", "content": "For authentication, you can use OAuth2 with Password (and hashing), Bearer with JWT tokens. For the database, SQLAlchemy or Tortoise ORM are popular. For background tasks, Celery or ARQ are good options."},
        {"role": "user", "content": "Let's start with the database. How do I set up SQLAlchemy with FastAPI?"},
        {"role": "assistant", "content": "You'll need to create a `database.py` file. First, install `sqlalchemy`. Then define your `SessionLocal` and `Base` classes. Here is a snippet:\n\n```python\nfrom sqlalchemy import create_engine\nfrom sqlalchemy.ext.declarative import declarative_base\nfrom sqlalchemy.orm import sessionmaker\n\nSQLALCHEMY_DATABASE_URL = 'sqlite:///./sql_app.db'\n\nengine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={'check_same_thread': False})\nSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)\nBase = declarative_base()\n```"},
        {"role": "user", "content": "Okay, what about the models?"},
        {"role": "assistant", "content": "You inherit from `Base`. For example:\n\n```python\nfrom sqlalchemy import Column, Integer, String\nfrom .database import Base\n\nclass User(Base):\n    __tablename__ = 'users'\n    id = Column(Integer, primary_key=True, index=True)\n    email = Column(String, unique=True, index=True)\n```"},
        # ... adding some noise/irrelevant context to test retrieval ...
        {"role": "user", "content": "By the way, what is the weather like in San Francisco?"},
        {"role": "assistant", "content": "I don't have real-time weather info, but it's usually foggy in the mornings!"},
        {"role": "user", "content": "Back to coding. How do I create a dependency for the DB session?"},
        {"role": "assistant", "content": "You can create a `get_db` function:\n\n```python\ndef get_db():\n    db = SessionLocal()\n    try:\n        yield db\n    finally:\n        db.close()\n```"},
        {"role": "user", "content": "Great. Now, how do I use this in a route?"},
        {"role": "assistant", "content": "You inject it into your path operation function using `Depends`."},
        {"role": "user", "content": "I'm getting an error: 'ModuleNotFoundError: No module named sqlalchemy'."},
        {"role": "assistant", "content": "You probably forgot to install it. Run `pip install sqlalchemy`."},
        {"role": "user", "content": "Thanks, that worked. Now about that background worker..."},
        {"role": "assistant", "content": "Right, for Celery, you need a message broker like Redis or RabbitMQ."},
    ]
    
    # Add timestamps
    for i, msg in enumerate(conversation):
        msg["ts"] = time.time() + i  # simple increment
        msg["source"] = "chat"

    print(f"    Generated {len(conversation)} messages.")

    # 3. Ingest
    print("\n[3] Ingesting Data...")
    start_ingest = time.time()
    optimizer.ingest(session_id, conversation)
    print(f"    Ingested in {time.time() - start_ingest:.4f} seconds.")

    # 4. Optimize (Retrieve)
    print("\n[4] Running Optimization...")
    query = "How do I configure Celery with Redis?"
    print(f"    Query: '{query}'")
    
    start_opt = time.time()
    result = optimizer.optimize(
        session_id=session_id,
        query_text=query,
        max_chunks=3, # Restrict to top 3 chunks to force shrinking
        target_token_budget=500 # Tight budget
    )
    duration = time.time() - start_opt
    print(f"    Optimization took {duration:.4f} seconds.")

    # 5. Output Results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    
    print(f"\nSTATS:")
    print(f"  - Raw Token Estimate (Full History): {result['raw_token_est']}")
    print(f"  - Optimized Token Count: {result['optimized_token_est']}")
    print(f"  - Tokens Reduced: {result['raw_token_est'] - result['optimized_token_est']}")
    print(f"  - Percent Saved: {result['percent_saved_est']:.1f}%")
    
    print("\n" + "-"*20)
    print("CONTEXT BEFORE (Full History - First 500 chars preview)")
    print("-" * 20)
    full_text = "\n".join([f"{m['role']}: {m['content']}" for m in conversation])
    print(full_text[:500] + "...\n[... truncated ...]")

    print("\n" + "-"*20)
    print("CONTEXT AFTER (Optimized/Shrunk)")
    print("-" * 20)
    for chunk in result["optimized_context"]:
        print(chunk["summary"])
        print("-" * 10)

    if duration < 2.0:
        print("\nSUCCESS: Optimization was under 2 seconds.")
    else:
        print(f"\nWARNING: Optimization took {duration:.2f} seconds (Target: < 2s).")

if __name__ == "__main__":
    main()
