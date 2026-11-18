import json
import logging
import time
from collections import defaultdict
from threading import Lock, RLock
from uuid import uuid4
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from config import (
    ALLOWED_ORIGINS,
    ENABLE_APP_LAUNCHER,
    ENABLE_LOCAL_FILESYSTEM,
    GEMINI_API_KEY,
    MAX_CONTEXT_MESSAGES,
    MAX_CONVERSATIONS,
    MAX_HISTORY_MESSAGES,
    MAX_MESSAGE_LENGTH,
    MAX_OUTPUT_TOKENS,
    MAX_REQUEST_SIZE_BYTES,
    MAX_TOOL_ROUNDS,
    MODEL,
    ORBIT_API_KEY,
    PORT,
    RATE_LIMIT_PER_MINUTE,
    TEMPERATURE,
    WORKSPACE,
)
from core.agent import Agent
from core.gemini_provider import GeminiProvider
from core.tool_registry import ToolRegistry
from tools.applications import launch_application, DEFINITION as APP_DEFINITION
from tools.filesystem import WorkspaceFiles, FILESYSTEM_DEFINITION
from tools.web_search import web_search, DEFINITION as WEB_SEARCH_DEFINITION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orbit")

class SimpleRateLimiter:
    """Sliding window rate limiter per client IP.

    NOTE: In serverless environments (such as Vercel Functions), this in-memory
    limiter only tracks requests within a single ephemeral function instance.
    It does not persist across cold starts or distribute across concurrent instances,
    so it does NOT provide production-grade abuse protection in serverless mode.
    For production abuse prevention on Vercel, use Vercel WAF / Firewall rules or
    an external distributed store like Upstash Redis.
    """
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def is_allowed(self, client_ip: str) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            cutoff = now - self.window_seconds
            timestamps = [t for t in self._requests[client_ip] if t > cutoff]
            if len(timestamps) >= self.max_requests:
                oldest = timestamps[0]
                retry_after = max(1, int(self.window_seconds - (now - oldest)) + 1)
                return False, retry_after
            timestamps.append(now)
            self._requests[client_ip] = timestamps

            # Clean stale IPs if table grows large
            if len(self._requests) > 1000:
                stale = [k for k, v in self._requests.items() if not v or v[-1] <= cutoff]
                for k in stale:
                    del self._requests[k]
            return True, 0

def create_app(agent=None, rate_limiter=None):
    app = Flask(__name__)
    app.url_map.strict_slashes = False
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_SIZE_BYTES

    # Configure CORS for public cloud deployment
    allowed_origins_raw = ALLOWED_ORIGINS.strip()
    if allowed_origins_raw == "*":
        cors_origins = "*"
    else:
        cors_origins = [orig.strip() for orig in allowed_origins_raw.split(",") if orig.strip()]

    CORS(
        app,
        resources={r"/api/*": {"origins": cors_origins}, r"/*": {"origins": cors_origins}},
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
        supports_credentials=True,
    )

    conversations: dict[str, list] = {}
    conversation_locks: dict[str, Lock] = {}
    state_lock = RLock()
    latency_samples: dict[str, list[dict]] = {"simple": [], "tools": []}
    if rate_limiter is None:
        rate_limiter = SimpleRateLimiter(max_requests=RATE_LIMIT_PER_MINUTE, window_seconds=60)
    app.rate_limiter = rate_limiter

    if agent is None:
        files = WorkspaceFiles(WORKSPACE)
        registry = ToolRegistry()
        registry.register(WEB_SEARCH_DEFINITION, web_search)
        registry.register(FILESYSTEM_DEFINITION, files.run)
        registry.register(APP_DEFINITION, launch_application)

        provider = GeminiProvider(api_key=GEMINI_API_KEY, model=MODEL)
        agent = Agent(
            provider=provider,
            registry=registry,
            model=MODEL,
            max_rounds=MAX_TOOL_ROUNDS,
            max_context_messages=MAX_CONTEXT_MESSAGES,
            options={"temperature": TEMPERATURE, "max_output_tokens": MAX_OUTPUT_TOKENS},
        )

    # Security headers on all responses
    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    # Global HTTP error handlers
    @app.errorhandler(413)
    def payload_too_large(error):
        return jsonify({"ok": False, "error": f"Request payload exceeds the {MAX_REQUEST_SIZE_BYTES // 1024} KB limit"}), 413

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"ok": False, "error": "Bad request"}), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"ok": False, "error": "Not found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"ok": False, "error": "Internal server error"}), 500

    def verify_api_protection():
        """Check optional ORBIT_API_KEY if configured in environment."""
        if not ORBIT_API_KEY:
            return None
        auth_header = request.headers.get("Authorization", "")
        key_header = request.headers.get("X-API-Key", "")
        provided_key = None
        if key_header:
            provided_key = key_header
        elif auth_header.startswith("Bearer "):
            provided_key = auth_header[7:].strip()

        if provided_key != ORBIT_API_KEY:
            return jsonify({"ok": False, "error": "Unauthorized: Invalid or missing API key"}), 401
        return None

    @app.get("/api/health")
    @app.get("/health")
    def health():
        has_key = bool(GEMINI_API_KEY) or hasattr(agent, "_fake") or agent is not None
        status = "ready" if has_key else "missing_api_key"
        return jsonify({
            "ok": True if (has_key or agent is not None) else False,
            "status": status,
            "provider": "gemini",
            "model": MODEL,
            "features": {
                "web_search": True,
                "cloud_mode": True,
                "local_filesystem": ENABLE_LOCAL_FILESYSTEM,
                "application_launcher": ENABLE_APP_LAUNCHER,
            },
        })

    @app.get("/api/metrics")
    @app.get("/metrics")
    def metrics():
        auth_err = verify_api_protection()
        if auth_err:
            return auth_err
        with state_lock:
            output = {}
            for mode, rows in latency_samples.items():
                values = sorted(row["total_ms"] for row in rows)
                output[mode] = {
                    "count": len(values),
                    "average_ms": round(sum(values) / len(values)) if values else None,
                    "p50_ms": values[len(values) // 2] if values else None,
                    "average_ttft_ms": (
                        round(
                            sum(row["ttft_ms"] for row in rows if row["ttft_ms"] is not None)
                            / len([row for row in rows if row["ttft_ms"] is not None])
                        )
                        if any(row["ttft_ms"] is not None for row in rows)
                        else None
                    ),
                    "recent": rows[-10:],
                }
        return jsonify({"ok": True, "latency": output})

    @app.post("/api/chat")
    @app.post("/chat")
    def chat():
        auth_err = verify_api_protection()
        if auth_err:
            return auth_err

        # Rate limiting check
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
        allowed, retry_after = app.rate_limiter.is_allowed(client_ip)
        if not allowed:
            return jsonify({
                "ok": False,
                "error": f"Rate limit exceeded. Please wait {retry_after} seconds before sending another message."
            }), 429

        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"ok": False, "error": "Request body must be a JSON object"}), 400
        message = body.get("message")
        if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_LENGTH:
            return jsonify({"ok": False, "error": f"message must be a non-empty string of at most {MAX_MESSAGE_LENGTH} characters"}), 400
        conversation_id = body.get("conversation_id") or str(uuid4())
        if not isinstance(conversation_id, str) or len(conversation_id) > 100:
            return jsonify({"ok": False, "error": "Invalid conversation_id"}), 400
        text = message.strip()
        client_history = body.get("history")

        with state_lock:
            # Enforce max conversation limit to prevent memory leak
            if len(conversations) >= MAX_CONVERSATIONS and conversation_id not in conversations:
                oldest_id = next(iter(conversations))
                del conversations[oldest_id]
                conversation_locks.pop(oldest_id, None)

            if isinstance(client_history, list) and client_history:
                # Sanitize and adopt client-provided history for serverless statelessness
                history = [
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in client_history
                    if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")
                ]
                conversations[conversation_id] = history
            else:
                history = conversations.setdefault(conversation_id, [])
            convo_lock = conversation_locks.setdefault(conversation_id, Lock())

        def events():
            with convo_lock:
                try:
                    for event in agent.stream_respond(history, text):
                        if event.get("type") == "done":
                            timing = event.get("timing", {})
                            mode = timing.get("mode", "simple")
                            with state_lock:
                                bucket = latency_samples.setdefault(mode, [])
                                bucket.append({
                                    "total_ms": timing.get("total_ms", 0),
                                    "model_ms": timing.get("model_ms", 0),
                                    "ttft_ms": timing.get("ttft_ms"),
                                })
                                del bucket[:-200]
                                if len(history) > MAX_HISTORY_MESSAGES:
                                    del history[:-MAX_HISTORY_MESSAGES]
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except Exception as exc:
                    # Remove a failed turn so retries do not duplicate the user's message.
                    if history and history[-1].get("role") == "user" and history[-1].get("content") == text:
                        history.pop()
                    app.logger.exception("Agent request failed")
                    err_str = str(exc)
                    if "GEMINI_API_KEY" in err_str:
                        user_error = "Gemini API key is not configured. Please set GEMINI_API_KEY in backend environment variables."
                    elif "429" in err_str or "quota" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str:
                        user_error = "Gemini API rate limit or quota exceeded. Please try again shortly."
                    elif "API_KEY_INVALID" in err_str or "401" in err_str or "403" in err_str:
                        user_error = "Invalid Gemini API key. Please check your GEMINI_API_KEY configuration."
                    else:
                        user_error = f"Agent unavailable: {exc.__class__.__name__}"
                    error = {"type": "error", "error": user_error}
                    yield f"data: {json.dumps(error)}\n\n"

        return Response(stream_with_context(events()), mimetype="text/event-stream", headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        })

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
