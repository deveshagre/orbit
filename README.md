# ORBIT

A cloud-ready personal AI agent with a responsive React chat UI, Flask + Gunicorn production API, Google Gemini tool calling, and live web search.

---

## Architecture

- **LLM Engine**: Google Gemini API via official `google-genai` SDK (default model: `gemini-2.5-flash`). Zero dependencies on local Ollama or GPU hardware.
- **Backend**: Flask 3 configured with production WSGI server **Gunicorn** (`gthread` multi-threaded worker supporting Server-Sent Events).
- **Frontend**: React + TypeScript + Vite with dynamic API target routing via `VITE_API_URL` and Single Page Application rewrite rules (`vercel.json`).
- **Security & Protection**:
  - `GEMINI_API_KEY` is loaded strictly by backend environment variables and never exposed to client browsers.
  - Strict Cross-Origin Resource Sharing (CORS) configurable via `ALLOWED_ORIGINS`.
  - In-memory sliding-window rate limiter per client IP.
  - Request body size limits (1 MB) and input length validation.
  - Local-only tools (desktop application launching and Mac filesystem access) are disabled by default in cloud deployments with informative explanations.
  - Web search via DuckDuckGo (`ddgs`) remains fully operational in cloud environments.

---

## Quick Setup (Local Development)

### 1. Requirements
- Python 3.10+
- Node.js 18+ and npm
- Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### 2. Backend Setup
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy example environment configuration
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY:
# GEMINI_API_KEY="your_api_key_here"
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

---

## Running Locally

### Development Mode

**Terminal 1 (Backend):**
```bash
cd backend
source .venv/bin/activate
python app.py
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`. The backend listens on `http://127.0.0.1:5100`.

### Production Emulation Mode (with Gunicorn)

```bash
cd backend
source .venv/bin/activate
gunicorn -c gunicorn.conf.py app:app
```

---

## Running Tests

Run the comprehensive 20-test automated test suite:
```bash
cd backend
source .venv/bin/activate
python -m pytest -v
```

---

## Cloud Deployment

ORBIT is designed for seamless zero-downtime deployment:
- **Frontend**: Deploy to **Vercel** (Root: `frontend`, Build: `npm run build`, Output: `dist`, Env: `VITE_API_URL`)
- **Backend**: Deploy to **Render** (Root: `backend`, Runtime: `Python 3`, Start: `gunicorn -c gunicorn.conf.py app:app`, Env: `GEMINI_API_KEY`, `ALLOWED_ORIGINS`)

See [DEPLOYMENT.md](DEPLOYMENT.md) for step-by-step instructions.

---

## Environment Variables

### Backend (`backend/.env` / Render)
| Variable | Default | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | *(None)* | **Required**. Google Gemini API key |
| `ORBIT_MODEL` | `gemini-2.5-flash` | Gemini model variant |
| `PORT` | `5100` | Server listening port (Render injects automatically) |
| `ALLOWED_ORIGINS` | `*` | Allowed CORS origins (e.g. `https://orbit-app.vercel.app`) |
| `ORBIT_API_KEY` | *(None)* | Optional shared secret token for header authentication |
| `ORBIT_RATE_LIMIT_PER_MINUTE` | `60` | Max requests per IP per minute |
| `ORBIT_ENABLE_APP_LAUNCHER` | `false` | Enable/disable desktop app launcher (disabled in cloud) |
| `ORBIT_ENABLE_LOCAL_FILESYSTEM` | `false` | Enable/disable workspace file access (disabled in cloud) |

### Frontend (`frontend/.env` / Vercel)
| Variable | Default | Description |
| :--- | :--- | :--- |
| `VITE_API_URL` | `""` | Target backend URL (e.g. `https://orbit-backend.onrender.com`) |
