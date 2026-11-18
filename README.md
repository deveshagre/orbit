# ORBIT

A modern, cloud-ready personal AI agent with a responsive React chat UI, Python Flask API on Vercel Serverless Functions, Google Gemini tool calling, and live web search. Hosted as a single project on Vercel.

---

## Architecture

- **Hosting**: Single-project, single-domain deployment on **Vercel**.
- **Frontend**: React + TypeScript + Vite, served from `frontend/dist`.
- **Backend API**: Python + Flask running as Vercel Serverless Functions via [`api/index.py`](file:///Users/devesh/Developer/orbit/api/index.py) with Server-Sent Events (SSE) streaming and up to 300s execution duration.
- **AI Engine**: Google Gemini API via official `google-genai` SDK (`gemini-2.5-flash`). Zero dependencies on local Ollama or GPU hardware.
- **Routing & Networking**: Single domain (e.g. `https://orbit.vercel.app`) with automatic same-origin routing for `/api/*` and SPA fallback for `/*`. No cross-origin CORS configuration required in production.
- **Security & Protection**:
  - `GEMINI_API_KEY` is loaded strictly by the serverless Python environment and is never exposed in browser or client code.
  - Desktop app launching and Mac filesystem access are disabled by default with informative explanations.
  - Live web search via DuckDuckGo (`ddgs`) remains fully operational in serverless environments.
  - Stateless multi-turn context passed via payload, avoiding reliance on ephemeral server memory.

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

# Configure environment variables
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

---

## Running Locally

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

Open `http://localhost:5173`. The backend listens on `http://127.0.0.1:5100` and is automatically proxied by Vite.

---

## Running Tests & Building

Run the 22-test automated backend test suite:
```bash
cd backend
source .venv/bin/activate
python -m pytest -v
```

Build the frontend from project root:
```bash
npm run build
```

---

## Vercel Deployment

Deploying ORBIT requires just one project on Vercel:
1. Import your GitHub repository into the **Vercel Dashboard**.
2. Keep the **Root Directory** as `./` (repository root).
3. Set `GEMINI_API_KEY` under **Project Settings > Environment Variables**.
4. Leave `VITE_API_URL` empty to use the same-domain `/api/*` routes.
5. Click **Deploy**.

See [DEPLOYMENT.md](DEPLOYMENT.md) for full dashboard settings and details.
