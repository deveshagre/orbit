# ORBIT Cloud Deployment Guide

This guide details the deployment of ORBIT as a decoupled cloud-ready system:
- **Frontend**: React + Vite + TypeScript deployed on **Vercel**
- **Backend**: Flask + Google Gemini API + Gunicorn (`gthread` worker) deployed on **Render**

---

## Architecture Overview

```
                      ┌───────────────────────────────┐
                      │    Client Browser / User      │
                      └───────▲───────────────┬───────┘
                              │               │
                     Static Assets       SSE Streaming
                     (HTML/JS/CSS)      /api/chat, /api/health
                              │               │
                ┌─────────────┴──────┐ ┌──────▼─────────────────────┐
                │   Vercel (Frontend)│ │     Render (Backend API)   │
                │                    │ │                            │
                │  • React + Vite    │ │  • Flask + Gunicorn        │
                │  • vercel.json     │ │  • GeminiProvider          │
                │  • VITE_API_URL    │ │  • ToolRegistry & Search   │
                └────────────────────┘ │  • CORS, Rate Limiter      │
                                       └──────────────┬─────────────┘
                                                      │
                                             google-genai SDK
                                             (GEMINI_API_KEY)
                                                      │
                                       ┌──────────────▼─────────────┐
                                       │    Google Gemini API       │
                                       │   (gemini-2.5-flash)       │
                                       └────────────────────────────┘
```

### Key Cloud Features & Guarantees
1. **API Key Security**: `GEMINI_API_KEY` is loaded strictly by the backend environment. It is **never** bundled or exposed to the browser.
2. **Streaming Support**: Gunicorn is configured with the `gthread` worker class and extended timeouts (120s) to support Server-Sent Events (SSE).
3. **CORS Isolation**: The backend restricts cross-origin calls to the approved frontend origin (`ALLOWED_ORIGINS`).
4. **Local Tools Safeguards**: Desktop app launching and Mac filesystem access are disabled by default with explanatory feedback if invoked. Web search (`ddgs`) remains fully operational.
5. **DDoS & Abuse Protection**: Includes an in-memory sliding-window IP rate limiter (default: 60 req/min), 1 MB payload limits, and optional `ORBIT_API_KEY` authentication.

---

## 1. Backend Deployment on Render

### Step 1: Create a Render Web Service
1. Log in to your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository containing the ORBIT project.

### Step 2: Configure Service Settings
| Field | Value | Notes |
| :--- | :--- | :--- |
| **Name** | `orbit-backend` | Or any unique name |
| **Region** | Oregon (US West) or closest region | Select region near users |
| **Branch** | `main` | Production branch |
| **Root Directory** | `backend` | **Important**: Set to `backend` |
| **Runtime** | `Python 3` | |
| **Build Command** | `pip install -r requirements.txt` | Installs Flask, google-genai, gunicorn, etc. |
| **Start Command** | `gunicorn -c gunicorn.conf.py app:app` | Uses production configuration |
| **Instance Type** | Free or Starter | 512 MB+ recommended |

### Step 3: Add Backend Environment Variables
In the Render dashboard under **Environment**:

| Key | Example Value | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | `AIzaSy...` | **Required**. Your Google Gemini API Key |
| `ALLOWED_ORIGINS` | `https://orbit-app.vercel.app` | **Required**. Your Vercel frontend URL (use `*` for initial setup) |
| `ORBIT_MODEL` | `gemini-2.5-flash` | Gemini model (options: `gemini-2.5-flash`, `gemini-2.0-flash`) |
| `PYTHON_VERSION` | `3.11.9` | Ensures stable runtime dependencies |
| `ORBIT_ENABLE_APP_LAUNCHER` | `false` | Disables macOS app launcher in cloud |
| `ORBIT_ENABLE_LOCAL_FILESYSTEM` | `false` | Disables local Mac filesystem access |
| `ORBIT_RATE_LIMIT_PER_MINUTE` | `60` | Max requests per client IP per minute |
| `ORBIT_API_KEY` | *(optional)* | Secret token to require on API requests |

### Step 4: Configure Health Check Path
Under **Advanced**:
- **Health Check Path**: `/api/health`

### Step 5: Deploy & Note Backend URL
Click **Create Web Service**. Once deployed, Render will assign a public URL:
`https://orbit-backend.onrender.com`

---

## 2. Frontend Deployment on Vercel

### Step 1: Import Project into Vercel
1. Log in to [Vercel Dashboard](https://vercel.com/dashboard).
2. Click **Add New...** > **Project**.
3. Select your GitHub repository.

### Step 2: Configure Build & Output Settings
| Field | Value |
| :--- | :--- |
| **Framework Preset** | `Vite` |
| **Root Directory** | Click *Edit* and select `frontend` |
| **Build Command** | `npm run build` |
| **Output Directory** | `dist` |
| **Install Command** | `npm install` |

### Step 3: Set Frontend Environment Variables
In the Vercel project configuration under **Environment Variables**:

| Key | Value | Description |
| :--- | :--- | :--- |
| `VITE_API_URL` | `https://orbit-backend.onrender.com` | **Required**. Your live Render backend URL (no trailing slash) |

> [!IMPORTANT]
> Never set `GEMINI_API_KEY` in Vercel or any `VITE_*` variable. Only the backend communicates with the Gemini API.

### Step 4: Deploy & Finalize CORS
1. Click **Deploy**.
2. Once deployed, Vercel gives you your frontend URL (e.g., `https://orbit-app.vercel.app`).
3. Return to the **Render Dashboard** for `orbit-backend` and ensure `ALLOWED_ORIGINS` contains your Vercel domain:
   ```env
   ALLOWED_ORIGINS=https://orbit-app.vercel.app
   ```

---

## 3. Local Verification Before Deploying

You can verify the cloud architecture locally before deploying to the public web:

### Terminal 1: Backend with Gunicorn
```bash
cd backend
source .venv/bin/activate
export GEMINI_API_KEY="your_api_key_here"  # Or add to backend/.env
export ALLOWED_ORIGINS="http://localhost:5173"
gunicorn -c gunicorn.conf.py app:app
```

Verify backend health:
```bash
curl -i http://localhost:5100/api/health
```

### Terminal 2: Frontend with Vite
```bash
cd frontend
export VITE_API_URL="http://127.0.0.1:5100"
npm run dev
```

Open `http://localhost:5173` in your browser. The header should display **Gemini 2.5 Flash**, the status pill should show **API ready**, and web searches will query through DuckDuckGo without any dependency on Ollama or local Mac files.

---

## 4. Environment Variables Reference

### Backend (`backend/.env` / Render)
| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | *(None)* | **Required**. Authentication key for Google Gemini API |
| `ORBIT_MODEL` | `gemini-2.5-flash` | Gemini model variant |
| `PORT` | `5100` | HTTP port (Render injects this dynamically) |
| `ALLOWED_ORIGINS` | `*` | Comma-separated list of allowed frontend origins |
| `ORBIT_API_KEY` | *(None)* | Optional token for basic API header protection |
| `ORBIT_RATE_LIMIT_PER_MINUTE` | `60` | Client IP sliding-window rate limit |
| `ORBIT_MAX_REQUEST_SIZE_BYTES` | `1048576` | Maximum JSON request size (1 MB) |
| `ORBIT_MAX_MESSAGE_LENGTH` | `8000` | Max characters per user prompt |
| `ORBIT_MAX_TOOL_ROUNDS` | `5` | Maximum agent function-calling rounds |
| `ORBIT_CONTEXT_MESSAGES` | `8` | Number of previous conversation turns preserved |
| `ORBIT_MAX_OUTPUT_TOKENS` | `1024` | Token limit for LLM responses |
| `ORBIT_TEMPERATURE` | `0.2` | Generation temperature |
| `ORBIT_ENABLE_APP_LAUNCHER` | `false` | Enable/disable macOS app launcher |
| `ORBIT_ENABLE_LOCAL_FILESYSTEM` | `false` | Enable/disable workspace filesystem tool |
| `WEB_CONCURRENCY` | `2` | Gunicorn worker process count |
| `GUNICORN_THREADS` | `4` | Gunicorn threads per worker |
| `GUNICORN_TIMEOUT` | `120` | Request timeout for long agent streaming |

### Frontend (`frontend/.env` / Vercel)
| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `VITE_API_URL` | `""` | Base URL of deployed backend. If omitted, uses Vite proxy. |

