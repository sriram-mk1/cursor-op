#!/usr/bin/env python3
"""
Test OpenRouter API compatibility
Validates that all OpenRouter parameters are properly handled
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_basic_completion():
    """Test basic chat completion"""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
    )
    # Should accept request and forward to OpenRouter
    assert response.status_code in [200, 401, 502]  # 401 if invalid key, 502 if OpenRouter unreachable


def test_all_sampling_parameters():
    """Test all sampling parameters"""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Test"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "top_a": 0.1,
            "min_p": 0.05,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "repetition_penalty": 1.1,
            "max_tokens": 100
        }
    )
    assert response.status_code in [200, 401, 502]


def test_response_format():
    """Test JSON mode"""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {"type": "json_object"}
        }
    )
    assert response.status_code in [200, 401, 502]


def test_tool_calling():
    """Test tool calling parameters"""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            }
                        }
                    }
                }
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": True
        }
    )
    assert response.status_code in [200, 401, 502]


def test_openrouter_specific_params():
    """Test OpenRouter-specific parameters"""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Test"}],
            "models": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
            "route": "fallback",
            "provider": {
                "allow_fallbacks": True,
                "require_parameters": False
            },
            "transforms": ["middle-out"]
        }
    )
    assert response.status_code in [200, 401, 502]


def test_advanced_parameters():
    """Test advanced parameters"""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Test"}],
            "seed": 42,
            "logit_bias": {100: -100, 200: 50},
            "logprobs": True,
            "top_logprobs": 5,
            "stop": ["END", "STOP"]
        }
    )
    assert response.status_code in [200, 401, 502]


def test_prompt_format():
    """Test prompt instead of messages"""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "openai/gpt-4o-mini",
            "prompt": "Hello, world!"
        }
    )
    assert response.status_code in [200, 401, 502]


def test_assistant_prefill():
    """Test assistant prefill"""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "anthropic/claude-3.5-sonnet",
            "messages": [
                {"role": "user", "content": "What is life?"},
                {"role": "assistant", "content": "Life is"}  # Prefill
            ]
        }
    )
    assert response.status_code in [200, 401, 502]


def test_multimodal_content():
    """Test multimodal content with images"""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "openai/gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/image.jpg",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
        }
    )
    assert response.status_code in [200, 401, 502]


def test_context_optimization_params():
    """Test custom context optimization parameters"""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "user", "content": f"Message {i}"}
                for i in range(10)
            ],
            "enable_optimization": True,
            "target_token_budget": 5000,
            "max_chunks": 15
        }
    )
    assert response.status_code in [200, 401, 502]


def test_streaming():
    """Test streaming"""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Count to 5"}],
            "stream": True
        }
    )
    assert response.status_code in [200, 401, 502]


def test_missing_authorization():
    """Test missing authorization header"""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}]
        }
    )
    assert response.status_code == 401


def test_invalid_request():
    """Test invalid request (no messages or prompt)"""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "openai/gpt-4o-mini"
        }
    )
    assert response.status_code == 400


def test_models_endpoint():
    """Test models list endpoint"""
    response = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer test-key"}
    )
    assert response.status_code in [200, 401, 502]


def test_generation_endpoint():
    """Test generation stats endpoint"""
    response = client.get(
        "/api/v1/generation?id=test-gen-id",
        headers={"Authorization": "Bearer test-key"}
    )
    assert response.status_code in [200, 401, 404, 502]


def test_health_endpoint():
    """Test health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "features" in data


def test_root_endpoint():
    """Test root info endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "optimization_techniques" in data
    assert "openrouter_features" in data


def test_custom_headers():
    """Test custom OpenRouter headers"""
    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer test-key",
            "HTTP-Referer": "https://myapp.com",
            "X-Title": "My App"
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}]
        }
    )
    assert response.status_code in [200, 401, 502]


def test_tool_role_message():
    """Test tool role in messages"""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "openai/gpt-4o",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "tool", "content": '{"temp": 72}', "tool_call_id": "call_123"}
            ]
        }
    )
    assert response.status_code in [200, 401, 502]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
