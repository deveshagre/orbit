# ORBIT Vercel Deployment Guide

ORBIT is configured for **single-project, single-domain hosting on Vercel**:
- **Frontend**: React + Vite + TypeScript served as static assets from `frontend/dist`
- **Backend**: Python + Flask running as Vercel Serverless Functions from `api/index.py`
- **AI Engine**: Google Gemini API via official `google-genai` SDK
- **Domain**: Single public domain (e.g. `https://orbit.vercel.app`) with automatic same-origin API routing

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │        Client Browser / User           │
                      └───────────────────▲────────────────────┘
                                          │
                                 Single Public Domain
                             (https://orbit.vercel.app)
                                          │
                        ┌─────────────────┴──────────────────┐
                        │      Vercel (Single Project)       │
                        │                                    │
                        │  Static Assets & SPA Rewrites      │
                        │  • /assets/* -> frontend/dist      │
                        │  • /*        -> index.html         │
                        │                                    │
                        │  Serverless Function               │
                        │  • /api/*    -> api/index.py       │
                        │     - Python runtime (WSGI/Flask)  │
                        │     - SSE Streaming (no-transform) │
                        │     - maxDuration: 300s (Hobby)    │
                        │     - google-genai SDK             │
                        └─────────────────┬──────────────────┘
                                          │
                                   GEMINI_API_KEY
                                (Backend Environment)
                                          │
                        ┌─────────────────▼──────────────────┐
                        │        Google Gemini API           │
                        │       (gemini-2.5-flash)           │
                        └────────────────────────────────────┘
```

---

## 1. Exact Vercel Dashboard Settings

When importing the repository into the [Vercel Dashboard](https://vercel.com/dashboard):

### Project Configuration
| Setting | Exact Value | Notes |
| :--- | :--- | :--- |
| **Project Name** | `orbit` | Or any preferred name |
| **Framework Preset** | `Vite` (or `Other`) | Vercel detects Vite or uses `vercel.json` |
| **Root Directory** | `./` (Leave as default `.`) | **Do not select `frontend`**. Must remain at repository root so Vercel discovers both `frontend/` and `api/`. |
| **Build Command** | `npm run build` | Defined in root `package.json` (`npm --prefix frontend install && npm --prefix frontend run build`) |
| **Output Directory** | `frontend/dist` | Pre-configured in `vercel.json` |
| **Install Command** | `npm install` | Installs root orchestrator script |

---

### Environment Variables
Under **Project Settings > Environment Variables** in Vercel:

| Key | Recommended Value | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | `AIzaSy...` | **Required**. Your Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/). |
| `ORBIT_MODEL` | `gemini-2.5-flash` | Gemini model variant (optional, defaults to `gemini-2.5-flash`). |
| `ORBIT_FALLBACK_MODELS` | `gemini-2.0-flash,gemini-1.5-flash` | Comma-separated list of confirmed fallback models if primary model is 503 overloaded. |
| `ORBIT_MAX_RETRIES` | `3` | Maximum retry attempts per model with bounded backoff and jitter. |
| `ORBIT_REQUEST_TIMEOUT_SECONDS` | `260` | Deadline cap terminating requests cleanly before Vercel 300s timeout. |
| `VITE_API_URL` | *(Leave empty)* | **Leave empty or unset**. When empty, the frontend calls `/api/*` on the same domain, eliminating CORS preflight. |
| `ALLOWED_ORIGINS` | `*` | Optional CORS header fallback. |
| `ORBIT_ENABLE_APP_LAUNCHER` | `false` | Disables macOS app launcher in cloud. |
| `ORBIT_ENABLE_LOCAL_FILESYSTEM` | `false` | Disables local Mac filesystem access. |

> [!IMPORTANT]
> `GEMINI_API_KEY` is loaded exclusively inside the serverless Python environment (`api/index.py`). It is never bundled into static assets or exposed to the browser.

---

## 2. Serverless Function Specifications

- **Entrypoint**: [`api/index.py`](file:///Users/devesh/Developer/orbit/api/index.py) exports the WSGI `app` callable from `backend/app.py`.
- **Dependencies**: Root [`requirements.txt`](file:///Users/devesh/Developer/orbit/requirements.txt), [`api/requirements.txt`](file:///Users/devesh/Developer/orbit/api/requirements.txt), and [`backend/requirements.txt`](file:///Users/devesh/Developer/orbit/backend/requirements.txt) are self-contained and identical, avoiding relative path resolution failures during Vercel's isolated container builds.
- **Max Duration**: Configured to `300` seconds in [`vercel.json`](file:///Users/devesh/Developer/orbit/vercel.json) (supported on Vercel Fluid Compute).
- **File Bundling**: [`vercel.json`](file:///Users/devesh/Developer/orbit/vercel.json) targets `"includeFiles": "backend/{*.py,core/**/*.py,tools/**/*.py}"` and explicitly excludes `.venv`, `__pycache__`, `.pytest_cache`, and `tests` via `"excludeFiles": "backend/{.venv/**,__pycache__/**,.pytest_cache/**,tests/**}"` to ensure lean lambda bundle sizes.
- **SSE Streaming**: Responses return `Cache-Control: no-cache, no-transform` and `X-Accel-Buffering: no`, allowing Server-Sent Events to stream tokens and live tool activity directly to the browser.
- **Routing Isolation**: [`vercel.json`](file:///Users/devesh/Developer/orbit/vercel.json) routes all `/api/(.*)` requests to `api/index.py` before applying the SPA fallback `/((?!api/).*)` to `index.html`. This ensures the single-page application never intercepts API calls.

---

## 3. Serverless State & Rate Limiting Behavior

1. **Stateless Multi-Turn Context**:
   Serverless functions are ephemeral and may be cold-started or routed to different instances across requests. ORBIT preserves conversation context by having the client provide previous conversation turns (`history`) in the request payload. The backend sanitizes and injects this history into the Gemini context window, ensuring coherent multi-turn reasoning without requiring a persistent database or local disk state.

2. **In-Memory Rate Limiting Disclaimer**:
   The backend includes an in-memory sliding-window limiter (`SimpleRateLimiter`). In a serverless environment like Vercel, this in-memory counter is scoped only to a single warm function instance and does not persist across cold starts or distribute across concurrent containers. **It does not provide production-grade abuse or DDoS protection.** For production abuse prevention, enable Vercel Attack Challenge Mode / WAF rules in the Vercel dashboard.

3. **Guarded Local Tools**:
   Opening desktop applications (`open -a`) and accessing the Mac filesystem are disabled. If a prompt triggers these actions, the agent returns an informative message explaining that the tool is unavailable in cloud/serverless hosting. Live web search via DuckDuckGo (`ddgs`) remains fully operational.

---

## 4. Local Development & Verification

### Running Locally
```bash
# Terminal 1: Backend
cd backend
source .venv/bin/activate
export GEMINI_API_KEY="your_api_key_here"
python app.py

# Terminal 2: Frontend
cd frontend
npm run dev
```

### Running Automated Tests
```bash
cd backend
source .venv/bin/activate
python -m pytest -v
```

### Building the Production Frontend
From project root:
```bash
npm run build
```
This runs `npm --prefix frontend install && npm --prefix frontend run build` and outputs the build to `frontend/dist/`.
