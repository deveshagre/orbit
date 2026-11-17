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
