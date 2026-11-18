import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Gemini Provider Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("ORBIT_MODEL", "gemini-2.5-flash")
FALLBACK_MODELS = [m.strip() for m in os.getenv("ORBIT_FALLBACK_MODELS", "gemini-2.0-flash,gemini-1.5-flash").split(",") if m.strip()]
MAX_RETRIES = int(os.getenv("ORBIT_MAX_RETRIES", "3"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("ORBIT_REQUEST_TIMEOUT_SECONDS", "260"))

# Server & Network Configuration
PORT = int(os.getenv("PORT", os.getenv("ORBIT_PORT", "5100")))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
ORBIT_API_KEY = os.getenv("ORBIT_API_KEY", "")

# Limits & Protection
RATE_LIMIT_PER_MINUTE = int(os.getenv("ORBIT_RATE_LIMIT_PER_MINUTE", "60"))
MAX_REQUEST_SIZE_BYTES = int(os.getenv("ORBIT_MAX_REQUEST_SIZE_BYTES", str(1024 * 1024)))  # 1 MB
MAX_MESSAGE_LENGTH = int(os.getenv("ORBIT_MAX_MESSAGE_LENGTH", "8000"))
MAX_CONVERSATIONS = int(os.getenv("ORBIT_MAX_CONVERSATIONS", "500"))

# Agent Context & Generation Controls
MAX_TOOL_ROUNDS = int(os.getenv("ORBIT_MAX_TOOL_ROUNDS", "5"))
MAX_HISTORY_MESSAGES = int(os.getenv("ORBIT_MAX_HISTORY_MESSAGES", "16"))
MAX_CONTEXT_MESSAGES = int(os.getenv("ORBIT_CONTEXT_MESSAGES", "8"))
MAX_OUTPUT_TOKENS = int(os.getenv("ORBIT_MAX_OUTPUT_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("ORBIT_TEMPERATURE", "0.2"))

# Local Tools Configuration (Disabled by default in cloud deployments)
ENABLE_APP_LAUNCHER = os.getenv("ORBIT_ENABLE_APP_LAUNCHER", "false").lower() in ("true", "1")
ENABLE_LOCAL_FILESYSTEM = os.getenv("ORBIT_ENABLE_LOCAL_FILESYSTEM", "false").lower() in ("true", "1")
WORKSPACE = Path(os.getenv("ORBIT_WORKSPACE", str(BASE_DIR.parent / "workspace"))).expanduser().resolve()
