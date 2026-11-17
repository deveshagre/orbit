import os

# Port binding: Render injects PORT; fallback to ORBIT_PORT or 5100
port = os.getenv("PORT", os.getenv("ORBIT_PORT", "5100"))
bind = f"0.0.0.0:{port}"

# Concurrency: gthread worker class enables multi-threaded concurrent SSE streaming
workers = int(os.getenv("WEB_CONCURRENCY", "2"))
threads = int(os.getenv("GUNICORN_THREADS", "4"))
worker_class = "gthread"

# Timeout: Allow long-running LLM and web search requests to complete without dropping SSE stream
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
keepalive = 5

# Logging: Stream to stdout/stderr for cloud log aggregators (e.g. Render, Datadog)
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")

