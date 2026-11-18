import json
from unittest.mock import MagicMock
import pytest
from app import create_app
from core.agent import Agent, requires_tools
from core.gemini_provider import GeminiProvider, GeminiStreamChunk
from core.tool_registry import ToolRegistry
from tools.applications import launch_application
from tools.filesystem import WorkspaceFiles
from google.genai import types
from google.genai.errors import APIError

class FakeProvider:
    """Mock provider simulating Gemini streaming chunks."""
    def __init__(self, chunk_batches):
        self.batches = iter(chunk_batches)
        self.calls = []
        self.model = "gemini-2.5-flash"

    def stream_chat(self, **kwargs):
        self.calls.append(kwargs)
        for chunk in next(self.batches):
            yield chunk

# -- 1. Tool Registry & Safety Tests --

def test_tool_registry_argument_validation():
    reg = ToolRegistry()
    reg.register({"function": {"name": "test_fn"}}, lambda a: {"ok": True})
    assert reg.execute("test_fn", "not_a_dict")["ok"] is False
    assert reg.execute("missing_fn", {})["ok"] is False

def test_filesystem_disabled_in_cloud_mode(monkeypatch):
    monkeypatch.delenv("ORBIT_ENABLE_LOCAL_FILESYSTEM", raising=False)
    fs = WorkspaceFiles("/tmp/orbit_test_disabled")
    result = fs.run({"action": "list"})
    assert result["ok"] is False
    assert "disabled in cloud deployments" in result["error"]
    assert "local files on your Mac" in result["error"]

def test_filesystem_containment_and_traversal_when_enabled(tmp_path):
    fs = WorkspaceFiles(tmp_path / "workspace", enabled=True)
    assert fs.run({"action": "read", "path": "../secret"})["ok"] is False
    assert fs.run({"action": "read", "path": str(tmp_path / "secret")})["ok"] is False

def test_file_create_does_not_overwrite_when_enabled(tmp_path):
    fs = WorkspaceFiles(tmp_path, enabled=True)
    first = fs.run({"action": "create_file", "path": "notes.txt", "content": "Hello ORBIT"})
    second = fs.run({"action": "create_file", "path": "notes.txt", "content": "changed"})
    assert first["ok"] is True
    assert second["ok"] is False
    assert (tmp_path / "notes.txt").read_text() == "Hello ORBIT"

def test_application_launcher_disabled_in_cloud_mode(monkeypatch):
    monkeypatch.delenv("ORBIT_ENABLE_APP_LAUNCHER", raising=False)
    result = launch_application({"application": "VS Code"})
    assert result["ok"] is False
    assert "disabled in cloud deployments" in result["error"]
    assert "local-only feature" in result["error"]

def test_application_allowlist_when_enabled(monkeypatch):
    monkeypatch.setenv("ORBIT_ENABLE_APP_LAUNCHER", "true")
    import tools.applications as apps
    monkeypatch.setattr(apps.platform, "system", lambda: "Darwin")
    seen = []
    monkeypatch.setattr(apps.subprocess, "Popen", lambda args, **kw: seen.append(args))
    assert launch_application({"application": "UnknownApp"})["ok"] is False
    result = launch_application({"application": "VS Code"})
    assert result["ok"] and seen == [["open", "-a", "Visual Studio Code"]]

# -- 2. Intent Routing --

def test_simple_prompts_bypass_tools():
    for prompt in ["Hi", "What is Python?", "Explain machine learning.", "Tell me a joke"]:
        assert not requires_tools(prompt)

def test_action_prompts_enter_tool_path():
    assert requires_tools("Search the web for Python 3.14 release notes")
    assert requires_tools("Open VS Code.")
    assert requires_tools("List files in my workspace.")

# -- 3. Gemini Provider Conversion & Agent Engine --

def test_gemini_provider_tool_conversion():
    provider = GeminiProvider(api_key="test_key")
    tool_defs = [
        {
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        }
    ]
    tools = provider.convert_tool_definitions(tool_defs)
    assert len(tools) == 1
    decl = tools[0].function_declarations[0]
    assert decl.name == "web_search"
    assert decl.description == "Search the web"

def test_gemini_provider_history_to_contents():
    provider = GeminiProvider(api_key="test_key")
    history = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a language."},
        {"role": "user", "content": "Search online for updates"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "web_search", "arguments": {"query": "python updates"}}]},
        {"role": "tool", "name": "web_search", "content": json.dumps({"ok": True, "summary": "Found 5 updates"})},
    ]
    contents = provider.history_to_contents(history)
    assert len(contents) == 5
    assert contents[0].role == "user"
    assert contents[1].role == "model"
    assert contents[2].role == "user"
    assert contents[3].role == "model"
    assert contents[4].role == "user"
    assert contents[4].parts[0].function_response.name == "web_search"

def test_simple_stream_direct_response():
    provider = FakeProvider([[
        GeminiStreamChunk(text="Hello "),
        GeminiStreamChunk(text="from Gemini!"),
    ]])
    agent = Agent(provider, ToolRegistry(), "gemini-2.5-flash")
    events = list(agent.stream_respond([], "Hi"))
    assert [e["text"] for e in events if e["type"] == "delta"] == ["Hello ", "from Gemini!"]
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == "Hello from Gemini!"
    assert events[-1]["timing"]["mode"] == "simple"
    assert "tools" in provider.calls[0]
    assert provider.calls[0]["tools"] is None  # simple prompt omits tools

def test_agent_stream_tool_flow():
    reg = ToolRegistry()
    reg.register(
        {"type": "function", "function": {"name": "echo", "description": "Echo back"}},
        lambda a: {"ok": True, "summary": f"echoed: {a.get('val')}"},
    )
    provider = FakeProvider([
        [GeminiStreamChunk(tool_calls=[{"name": "echo", "arguments": {"val": "cloud"}}])],
        [GeminiStreamChunk(text="Result is cloud.")],
    ])
    history = []
    agent = Agent(provider, reg, "gemini-2.5-flash")
    events = list(agent.stream_respond(history, "search online for cloud"))

    assert any(e["type"] == "activity" for e in events)
    activities = [e["activity"] for e in events if e["type"] == "activity" and e["activity"].get("status") == "done"]
    assert len(activities) == 1
    assert activities[0]["tool"] == "echo"
    assert activities[0]["ok"] is True
    assert events[-1]["answer"] == "Result is cloud."
    assert events[-1]["timing"]["mode"] == "tools"

def test_context_window_bounding():
    history = []
    for i in range(20):
        history.extend([{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}])
    provider = FakeProvider([[GeminiStreamChunk(text="Ack.")]])
    agent = Agent(provider, ToolRegistry(), "gemini-2.5-flash", max_context_messages=8)
    agent.respond(history, "new follow up")
    passed_history = provider.calls[0]["history"]
    assert len(passed_history) <= 9  # at most 8 context messages + current user message
    assert passed_history[-1]["content"] == "new follow up"
    assert passed_history[0]["role"] == "user"

# -- 4. API, Security & Hardening Tests --

def test_health_endpoint():
    app = create_app(agent=object())
    client = app.test_client()
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["ok"] is True
    assert data["provider"] == "gemini"
    assert "model" in data
    assert data["features"]["cloud_mode"] is True

def test_latency_metrics_endpoint():
    app = create_app(agent=object())
    client = app.test_client()
    res = client.get("/api/metrics")
    assert res.status_code == 200
    assert set(res.get_json()["latency"].keys()) == {"simple", "tools"}

def test_missing_gemini_api_key_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiProvider(api_key="")
    agent = Agent(provider, ToolRegistry(), "gemini-2.5-flash")
    app = create_app(agent=agent)
    client = app.test_client()
    res = client.post("/api/chat", json={"message": "Hello"})
    assert res.status_code == 200
    assert b"Gemini API key is not configured" in res.data

def test_cors_headers():
    app = create_app(agent=object())
    client = app.test_client()
    res = client.get("/api/health", headers={"Origin": "https://orbit-app.vercel.app"})
    assert res.status_code == 200
    assert "Access-Control-Allow-Origin" in res.headers

def test_rate_limiting():
    from app import SimpleRateLimiter
    provider = FakeProvider([[GeminiStreamChunk(text="ok")] for _ in range(5)])
    agent = Agent(provider, ToolRegistry(), "gemini-2.5-flash")
    limited_limiter = SimpleRateLimiter(max_requests=2, window_seconds=60)
    app = create_app(agent=agent, rate_limiter=limited_limiter)
    client = app.test_client()

    r1 = client.post("/api/chat", json={"message": "msg 1"})
    r1.get_data()
    r2 = client.post("/api/chat", json={"message": "msg 2"})
    r2.get_data()
    r3 = client.post("/api/chat", json={"message": "msg 3"})
    r3.get_data()

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429
    assert "Rate limit exceeded" in r3.get_json()["error"]

def test_payload_size_limit():
    app = create_app(agent=object())
    client = app.test_client()
    # Sending payload exceeding MAX_CONTENT_LENGTH (1 MB)
    large_payload = "a" * (1024 * 1024 + 500)
    res = client.post("/api/chat", data=large_payload, content_type="application/json")
    assert res.status_code == 413

def test_optional_api_key_protection(monkeypatch):
    monkeypatch.setattr("app.ORBIT_API_KEY", "secret_token_123")
    app = create_app(agent=object())
    client = app.test_client()

    # Without key -> 401
    r_unauth = client.post("/api/chat", json={"message": "Hello"})
    assert r_unauth.status_code == 401

    # With bad key -> 401
    r_bad = client.post("/api/chat", json={"message": "Hello"}, headers={"X-API-Key": "wrong"})
    assert r_bad.status_code == 401

    # With valid key in header -> allowed through
    r_ok = client.post("/api/chat", json={"message": "Hello"}, headers={"X-API-Key": "secret_token_123"})
    r_ok.get_data()
    assert r_ok.status_code == 200

def test_vercel_function_entrypoint_and_aliases():
    import sys
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    import api.index as vercel_entrypoint
    assert hasattr(vercel_entrypoint, "app")
    client = vercel_entrypoint.app.test_client()
    res1 = client.get("/api/health")
    assert res1.status_code == 200
    res2 = client.get("/health")
    assert res2.status_code == 200
    res3 = client.get("/api/health/")
    assert res3.status_code == 200
    # Verify post routes match both prefixes
    r_chat1 = client.post("/api/chat", json={"message": "ping"})
    assert r_chat1.status_code in (200, 429)
    r_chat1.get_data()
    r_chat2 = client.post("/chat", json={"message": "ping"})
    assert r_chat2.status_code in (200, 429)
    r_chat2.get_data()

def test_client_provided_history_stateless():
    provider = FakeProvider([[GeminiStreamChunk(text="Context acknowledged.")]])
    agent = Agent(provider, ToolRegistry(), "gemini-2.5-flash")
    app = create_app(agent=agent)
    client = app.test_client()

    history_payload = [
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Nice to meet you, Alice."},
    ]
    res = client.post("/api/chat", json={
        "message": "What is my name?",
        "conversation_id": "test-stateless",
        "history": history_payload
    })
    res.get_data()
    assert res.status_code == 200
    passed_history = provider.calls[0]["history"]
    assert any(m["content"] == "My name is Alice" for m in passed_history)

# -- 5. Reliability, Retry, Fallback & Error Handling Tests --

def test_gemini_provider_retry_on_503_success():
    sleeps = []
    mock_client = MagicMock()
    # Attempt 1 raises 503; Attempt 2 succeeds
    chunk = GeminiStreamChunk(text="Success on retry")
    err_503 = APIError(503, {"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}})

    attempt_count = 0
    def mock_generate_stream(*args, **kwargs):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise err_503
        return iter([chunk])

    mock_client.models.generate_content_stream.side_effect = mock_generate_stream

    provider = GeminiProvider(
        api_key="test_key",
        model="gemini-2.5-flash",
        client=mock_client,
        max_attempts=3,
        backoff_delays=(2.0, 5.0, 10.0),
        sleep_func=lambda s: sleeps.append(s),
    )

    chunks = list(provider.stream_chat(
        history=[{"role": "user", "content": "ping"}],
        system_instruction="sys",
    ))
    assert len(chunks) == 1
    assert chunks[0].text == "Success on retry"
    assert attempt_count == 2
    assert len(sleeps) == 1
    # Bounded delay of ~2s with jitter (1.7s to 2.3s)
    assert 1.6 <= sleeps[0] <= 2.4

def test_gemini_provider_non_retryable_errors_fail_immediately():
    for status_code, status_str in [(401, "UNAUTHENTICATED"), (403, "PERMISSION_DENIED"), (404, "NOT_FOUND"), (429, "RESOURCE_EXHAUSTED")]:
        sleeps = []
        mock_client = MagicMock()
        err = APIError(status_code, {"error": {"code": status_code, "message": "Non-retryable", "status": status_str}})
        mock_client.models.generate_content_stream.side_effect = err

        provider = GeminiProvider(
            api_key="test_key",
            model="gemini-2.5-flash",
            client=mock_client,
            max_attempts=3,
            sleep_func=lambda s: sleeps.append(s),
        )

        with pytest.raises(APIError) as exc_info:
            list(provider.stream_chat(
                history=[{"role": "user", "content": "ping"}],
                system_instruction="sys",
            ))
        assert exc_info.value.code == status_code
        assert len(sleeps) == 0  # Zero retries for non-retryable errors
        assert mock_client.models.generate_content_stream.call_count == 1

def test_gemini_provider_model_fallback_on_exhausted_retries():
    sleeps = []
    mock_client = MagicMock()

    # Confirmed models via API
    confirmed_model_objs = [
        types.Model(name="models/gemini-2.5-flash", supported_actions=["generateContent"]),
        types.Model(name="models/gemini-2.0-flash", supported_actions=["generateContent"]),
    ]
    mock_client.models.list.return_value = confirmed_model_objs

    # 2.5-flash fails on all 3 attempts with 503; 2.0-flash succeeds on attempt 1
    calls_by_model = []
    def mock_generate_stream(*args, **kwargs):
        m = kwargs.get("model")
        calls_by_model.append(m)
        if m == "gemini-2.5-flash":
            raise APIError(503, {"error": {"code": 503, "message": "High server demand", "status": "UNAVAILABLE"}})
        return iter([GeminiStreamChunk(text="Fallback model answered")])

    mock_client.models.generate_content_stream.side_effect = mock_generate_stream

    provider = GeminiProvider(
        api_key="test_key",
        model="gemini-2.5-flash",
        fallback_models=["gemini-2.0-flash", "invented-fake-model"],
        client=mock_client,
        max_attempts=3,
        sleep_func=lambda s: sleeps.append(s),
    )

    chunks = list(provider.stream_chat(
        history=[{"role": "user", "content": "ping"}],
        system_instruction="sys",
    ))

    assert len(chunks) == 1
    assert chunks[0].text == "Fallback model answered"
    # Primary model called 3 times, fallback model called once
    assert calls_by_model.count("gemini-2.5-flash") == 3
    assert calls_by_model.count("gemini-2.0-flash") == 1
    # Invented fake model was never confirmed, so it was never called
    assert "invented-fake-model" not in calls_by_model

def test_gemini_provider_sanitizes_credentials_in_logs():
    secret_key = "AIzaSyTestSecretKey_123456789"
    provider = GeminiProvider(api_key=secret_key, client=MagicMock())
    raw_error = f"API error occurred: key={secret_key} with Bearer {secret_key}"
    sanitized = provider._sanitize(raw_error)
    assert secret_key not in sanitized
    assert "[REDACTED" in sanitized

def test_gemini_provider_timeout_budget_terminates_cleanly():
    mock_client = MagicMock()
    err_503 = APIError(503, {"error": {"code": 503, "message": "Overloaded", "status": "UNAVAILABLE"}})
    mock_client.models.generate_content_stream.side_effect = err_503

    provider = GeminiProvider(
        api_key="test_key",
        client=mock_client,
        max_attempts=3,
        request_timeout_seconds=0.001,  # immediate deadline
        sleep_func=lambda s: None,
    )

    with pytest.raises(TimeoutError) as exc_info:
        list(provider.stream_chat(
            history=[{"role": "user", "content": "ping"}],
            system_instruction="sys",
        ))
    assert "time budget exceeded" in str(exc_info.value).lower()

def test_sse_error_event_formatting_503():
    class FailingProvider:
        model = "gemini-2.5-flash"
        def stream_chat(self, **kwargs):
            raise APIError(503, {"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}})

    agent = Agent(FailingProvider(), ToolRegistry(), "gemini-2.5-flash")
    app = create_app(agent=agent)
    client = app.test_client()
    res = client.post("/api/chat", json={"message": "hello"})
    assert res.status_code == 200
    events = [json.loads(line.replace("data: ", "")) for line in res.data.decode().splitlines() if line.startswith("data: ")]
    assert any(e.get("type") == "error" and e.get("code") == 503 and "temporarily overloaded" in e.get("error", "") for e in events)

def test_sse_error_event_formatting_429():
    class QuotaFailingProvider:
        model = "gemini-2.5-flash"
        def stream_chat(self, **kwargs):
            raise APIError(429, {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}})

    agent = Agent(QuotaFailingProvider(), ToolRegistry(), "gemini-2.5-flash")
    app = create_app(agent=agent)
    client = app.test_client()
    res = client.post("/api/chat", json={"message": "hello"})
    assert res.status_code == 200
    events = [json.loads(line.replace("data: ", "")) for line in res.data.decode().splitlines() if line.startswith("data: ")]
    assert any(e.get("type") == "error" and e.get("code") == 429 and "rate limit or quota exceeded" in e.get("error", "").lower() for e in events)

def test_sse_error_event_formatting_401_403():
    class AuthFailingProvider:
        model = "gemini-2.5-flash"
        def stream_chat(self, **kwargs):
            raise APIError(401, {"error": {"code": 401, "message": "API key invalid", "status": "UNAUTHENTICATED"}})

    agent = Agent(AuthFailingProvider(), ToolRegistry(), "gemini-2.5-flash")
    app = create_app(agent=agent)
    client = app.test_client()
    res = client.post("/api/chat", json={"message": "hello"})
    assert res.status_code == 200
    events = [json.loads(line.replace("data: ", "")) for line in res.data.decode().splitlines() if line.startswith("data: ")]
    assert any(e.get("type") == "error" and e.get("code") == 401 and "authentication or permission" in e.get("error", "").lower() for e in events)
