#!/usr/bin/env python3
"""
Context Optimizer Gateway - Examples
Demonstrates all OpenRouter API features with context optimization
"""

import os
from openai import OpenAI

# Initialize client (works with any OpenAI-compatible client)
client = OpenAI(
    base_url="https://cursor-op.onrender.com/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "sk-or-v1-...")
)


def example_basic():
    """Basic chat completion"""
    print("=== Basic Example ===")
    
    response = client.chat.completions.create(
        model="anthropic/claude-3.5-sonnet",
        messages=[
            {"role": "user", "content": "What is BM25?"}
        ]
    )
    
    print(f"Response: {response.choices[0].message.content}\n")


def example_with_parameters():
    """Using sampling parameters"""
    print("=== Sampling Parameters ===")
    
    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {"role": "system", "content": "You are a creative writer."},
            {"role": "user", "content": "Write a haiku about code optimization"}
        ],
        temperature=0.9,
        max_tokens=100,
        top_p=0.95,
        frequency_penalty=0.5,
        presence_penalty=0.3
    )
    
    print(f"Haiku:\n{response.choices[0].message.content}\n")


def example_json_mode():
    """JSON mode for structured outputs"""
    print("=== JSON Mode ===")
    
    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {"role": "system", "content": "You return valid JSON only."},
            {"role": "user", "content": "List 3 programming languages with their use cases"}
        ],
        response_format={"type": "json_object"}
    )
    
    print(f"JSON Response:\n{response.choices[0].message.content}\n")


def example_streaming():
    """Streaming responses"""
    print("=== Streaming Example ===")
    
    stream = client.chat.completions.create(
        model="anthropic/claude-3.5-sonnet",
        messages=[
            {"role": "user", "content": "Count from 1 to 5 slowly"}
        ],
        stream=True
    )
    
    print("Streaming response: ", end="")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def example_tool_calling():
    """Function calling (tool use)"""
    print("=== Tool Calling ===")
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price for a company",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., AAPL)"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {"role": "user", "content": "What's the stock price of Apple?"}
        ],
        tools=tools,
        tool_choice="auto"
    )
    
    if response.choices[0].message.tool_calls:
        print(f"Tool calls: {response.choices[0].message.tool_calls}\n")
    else:
        print(f"Response: {response.choices[0].message.content}\n")


def example_context_optimization():
    """Context optimization with long conversation"""
    print("=== Context Optimization ===")
    
    # Simulate a long conversation (optimization kicks in at 4+ messages)
    messages = [
        {"role": "user", "content": "My name is Alice and I love Python programming."},
        {"role": "assistant", "content": "Nice to meet you, Alice! Python is a great language."},
        {"role": "user", "content": "I'm working on a machine learning project using scikit-learn."},
        {"role": "assistant", "content": "Scikit-learn is excellent for ML. What kind of model are you building?"},
        {"role": "user", "content": "A classification model for predicting customer churn."},
        {"role": "assistant", "content": "Customer churn prediction is a common and important use case. Are you using logistic regression or something more complex?"},
        {"role": "user", "content": "I'm trying random forests and gradient boosting."},
        {"role": "assistant", "content": "Great choices! Both are powerful ensemble methods."},
        {"role": "user", "content": "What was my name again?"}  # Testing context retention
    ]
    
    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=messages,
        extra_body={
            "enable_optimization": True,
            "max_chunks": 15,
            "target_token_budget": 8000
        }
    )
    
    print(f"Response (with optimization): {response.choices[0].message.content}")
    print(f"Note: Check response headers for optimization stats\n")


def example_assistant_prefill():
    """Assistant prefill to guide responses"""
    print("=== Assistant Prefill ===")
    
    response = client.chat.completions.create(
        model="anthropic/claude-3.5-sonnet",
        messages=[
            {"role": "user", "content": "What is the meaning of life?"},
            {"role": "assistant", "content": "I believe the meaning of life is"}  # Prefill
        ]
    )
    
    print(f"Prefilled response: {response.choices[0].message.content}\n")


def example_multimodal():
    """Image inputs (vision models)"""
    print("=== Multimodal (Vision) ===")
    
    response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        }
                    }
                ]
            }
        ]
    )
    
    print(f"Vision response: {response.choices[0].message.content}\n")


def example_custom_optimization():
    """Fine-tuned optimization settings"""
    print("=== Custom Optimization Settings ===")
    
    # Create a very long conversation
    messages = [
        {"role": "user", "content": f"This is message {i}: " + "x" * 100}
        for i in range(20)
    ]
    messages.append({"role": "user", "content": "Summarize our conversation"})
    
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=messages,
        extra_body={
            "enable_optimization": True,
            "target_token_budget": 2000,  # Strict budget
            "max_chunks": 5                # Keep only 5 most relevant chunks
        }
    )
    
    print(f"Response with aggressive optimization: {response.choices[0].message.content}\n")


def example_deterministic():
    """Deterministic outputs with seed"""
    print("=== Deterministic Outputs ===")
    
    for i in range(3):
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "user", "content": "Pick a random number between 1 and 100"}
            ],
            seed=42,  # Same seed = same output
            temperature=0
        )
        print(f"Attempt {i+1}: {response.choices[0].message.content}")
    
    print()


def example_disable_optimization():
    """Disable optimization when not needed"""
    print("=== Optimization Disabled ===")
    
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Hello!"}
        ],
        extra_body={
            "enable_optimization": False  # Disable for short conversations
        }
    )
    
    print(f"Response without optimization: {response.choices[0].message.content}\n")


if __name__ == "__main__":
    print("Context Optimizer Gateway - Examples\n")
    print("Make sure to set OPENROUTER_API_KEY environment variable\n")
    
    examples = [
        ("Basic", example_basic),
        ("Parameters", example_with_parameters),
        ("JSON Mode", example_json_mode),
        ("Streaming", example_streaming),
        ("Tool Calling", example_tool_calling),
        ("Context Optimization", example_context_optimization),
        ("Assistant Prefill", example_assistant_prefill),
        ("Multimodal", example_multimodal),
        ("Custom Optimization", example_custom_optimization),
        ("Deterministic", example_deterministic),
        ("Disabled Optimization", example_disable_optimization),
    ]
    
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print("\nRunning all examples...\n")
    print("=" * 80)
    print()
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {name}: {e}\n")
    
    print("=" * 80)
    print("\nAll examples completed!")
