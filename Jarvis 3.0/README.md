# A.O.R.I.S. — Multi-Model AI Orchestration System
**Team: APEX LEGENDS | Nebius × NVIDIA Global AI Hackathon 2026**

A.O.R.I.S. is an advanced, persistent-memory, multi-model AI assistant powered by **NVIDIA Nemotron** as its master orchestrator and final synthesis model, coordinating specialized reasoning, coding, and vision models alongside a native ReportLab document generation engine.

---

## 🏛️ System Architecture

A.O.R.I.S. employs a hierarchical orchestration topology: **NVIDIA Nemotron** coordinates all user interactions, performs intent decomposition, dispatches subtasks to domain specialist models when available, and executes **Final Synthesis** to review, validate, and refine specialist drafts into authoritative responses.

```
                              ┌─────────────────────────────────────────┐
                              │           USER / WEB UI / CLI           │
                              └────────────────────┬────────────────────┘
                                                   │
                                                   ▼
                              ┌─────────────────────────────────────────┐
                              │     A.O.R.I.S. FASTAPI BACKEND GATE     │
                              │  (API Auth, Rate Limiting, Safety Guard)│
                              └────────────────────┬────────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           NVIDIA NEMOTRON MASTER ORCHESTRATOR                                   │
│                        (via Nebius Token Factory / NVIDIA NIM API)                              │
│                                                                                                 │
│  ┌───────────────────────┐   ┌──────────────────────────┐   ┌────────────────────────────────┐  │
│  │   Agent Router        │   │  Multi-Model Registry    │   │  Persistent Memory Matrix      │  │
│  │ (Intent Classification│   │ (Status Auditing, Health │   │ (SQLite + FTS5 BM25 Retrieval, │  │
│  │  & Capability Match)  │   │  Fallback Resolution)    │   │  Provenance & Supersession)    │  │
│  └───────────┬───────────┘   └────────────┬─────────────┘   └────────────────────────────────┘  │
│              │                            │                                                     │
│              ▼                            ▼                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                 MULTI-MODEL SPECIALISTS                                   │  │
│  │                                                                                           │  │
│  │  ┌────────────────────────┐  ┌────────────────────────┐  ┌─────────────────────────────┐  │  │
│  │  │   OpenAI GPT           │  │   Anthropic Claude     │  │   Google Gemini             │  │  │
│  │  │   Reasoning Specialist │  │   Coding Specialist    │  │   Vision/Creative Specialist│  │  │
│  │  │ (Deep Deductive Logic) │  │ (Software Engineering) │  │  (Visual & Multimodal UI)   │  │  │
│  │  └───────────┬────────────┘  └───────────┬────────────┘  └──────────────┬──────────────┘  │  │
│  └──────────────┼───────────────────────────┼──────────────────────────────┼─────────────────┘  │
│                 └───────────────────────────┼──────────────────────────────┘                    │
│                                             ▼                                                   │
│                               ┌───────────────────────────┐                                     │
│                               │  NVIDIA NEMOTRON          │                                     │
│                               │  FINAL SYNTHESIS ENGINE   │                                     │
│                               │ (Fact-check, Tone, Polish)│                                     │
│                               └─────────────┬─────────────┘                                     │
└─────────────────────────────────────────────┼───────────────────────────────────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
      ┌──────────────────────────────┐                  ┌──────────────────────────────┐
      │     REPORTLAB PDF ENGINE     │                  │     GUARDED TOOLS SUITE      │
      │  Native zero-dep PDF reports │                  │  Calculator (Safe AST),      │
      │  Two-pass numbered canvas    │                  │  Sandboxed File System,      │
      │  /api/downloads/{file_name}  │                  │  Consequential Safety Gates  │
      └──────────────────────────────┘                  └──────────────────────────────┘
```

---

## 🌟 Key Capabilities & Differentiators

### 1. Master Orchestration & Final Synthesis (NVIDIA Nemotron)
- **Master Router**: Fast deterministic intent classification routes queries to general dialogue, multi-step planning, coding, deep reasoning, or document generation.
- **Specialist Dispatch**: Dispatches sub-tasks to specialized models (OpenAI for logic, Anthropic for software development, Google Gemini for visual/creative layouts).
- **Nemotron Final Synthesis**: The orchestrator reviews specialist draft outputs, checks for alignment with the user's intent, validates consistency against persistent memory, and synthesizes the final response.
- **Honest Fallback Guarantee**: If any specialist is unconfigured or credentials are missing, the system gracefully routes to Nemotron / deterministic offline baseline with `specialist_fallback=True`. The application never crashes or hides configuration status.

### 2. Standardized Multi-Model Status Auditing
Every model provider in the registry reports one of 5 standardized states:
- `AVAILABLE_VERIFIED`: Live credentials verified via active ping.
- `CREDENTIALS_MISSING`: API key not provided in environment.
- `AUTH_FAILED`: Invalid API key or permission denied (HTTP 401/403).
- `RATE_LIMITED`: Upstream provider rate limit reached (HTTP 429).
- `DISABLED`: Explicitly deactivated by administrator.

### 3. Pure-Python ReportLab PDF Generation
- **Zero External Dependencies**: Generates professional PDF reports natively using pure-Python ReportLab (no WeasyPrint, wkhtmltopdf, or browser runtime required).
- **Two-Pass `_NumberedCanvas`**: Computes accurate dynamic "Page X of Y" footers and timestamped headers.
- **Structural Disk Verification**: Enforces non-zero byte check, `%PDF-` header validation, and `%%EOF` trailer verification.
- **Secure Download Endpoint**: Serves generated documents via `/api/downloads/{file_name}` with strict path-traversal rejection (`..`, `/`, `\` blocked).

### 4. Persistent Memory Matrix (SQLite + FTS5)
- Tracks 4 conceptual memory layers: **Working**, **Episodic**, **Semantic**, and **Procedural**.
- Full-text search (BM25) via SQLite FTS5 across facts, tags, categories, and entities.
- Intelligent conflict resolution: detects contradictory updates (e.g., changes in user preferences) and automatically sets `superseded_by` pointers, preventing context bloat without loss of audit history.
- Provenance tracking with creation, update, and access timestamps, plus epistemic status (`confirmed_fact`, `user_claim`, `system_inference`).
- Soft-delete (`is_deleted=1`) and permanent purge capabilities.

### 5. Guarded Tool Execution & Safety Layer
- Permission levels: `READ_ONLY`, `CONSEQUENTIAL`, `DESTRUCTIVE`.
- **User Confirmation Gate**: Consequential operations (such as file writes) enter `AWAITING_CONFIRMATION` until explicitly approved or denied by the user.
- **Path Traversal Protection**: Rejects `..` escapes or unauthorized paths outside permitted directories.
- **Safe AST Evaluation**: Computes arithmetic via AST parsing without unsafe `eval()` or shell exposure.
- **Secret Redaction**: Redacts API keys and sensitive tokens before writing to terminal, logs, or UI.

---

## 🚀 Quickstart & Launch Instructions

### 1. Prerequisites
- Python 3.11+
- `uv` (recommended) or standard `pip`

### 2. Environment Setup
```powershell
# Clone or open workspace directory
cd "e:\Jarvis 3.0"

# Create virtual environment and install dependencies
uv venv .venv
uv pip install -e ".[dev]"
```

### 3. Configuration
Copy `.env.example` to `.env` and configure your credentials:
```powershell
cp .env.example .env
```

Key configuration parameters:
- **Primary Orchestrator**:
  - `NEBIUS_API_KEY`: API key for Nebius Token Factory (`https://api.studio.nebius.ai/v1`).
  - `NEBIUS_DEFAULT_MODEL`: Eligible open model on Nebius (e.g., `nvidia/nemotron-3-super-120b-a12b`).
  - `NVIDIA_API_KEY`: Fallback key for NVIDIA NIM (`https://integrate.api.nvidia.com/v1`).
- **Domain Specialists (Optional)**:
  - `OPENAI_API_KEY` / `OPENAI_MODEL`: OpenAI Reasoning Specialist (`gpt-4o-mini` or `o1`).
  - `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL`: Anthropic Coding Specialist (`claude-3-5-sonnet-20241022`).
  - `GOOGLE_API_KEY` / `GEMINI_MODEL`: Google Gemini Vision/Creative Specialist (`gemini-1.5-flash`).
- **Document Engine**:
  - `AORIS_DOCS_DIR`: Destination directory for generated PDF documents (default `data/generated_docs`).

> **Offline Fallback Guarantee**: If no remote API keys are set, A.O.R.I.S. automatically engages its **Offline Deterministic Provider**, allowing all multi-agent workflows, tool safety checks, memory operations, and UI interactions to function 100% locally without external network dependencies.

### 4. Running the Application

#### Option A: Launch Desktop Web UI & API Server
```powershell
.\.venv\Scripts\python -m jarvis.main serve --host 127.0.0.1 --port 8000
```
Open your browser to: **`http://127.0.0.1:8000`**

#### Option B: Command-Line Interface (CLI)
```powershell
# System & multi-model health audit
.\.venv\Scripts\python -m jarvis.main health

# Interactive chat with automatic specialist routing
.\.venv\Scripts\python -m jarvis.main chat "Write code for a thread-safe LRU cache"

# Explicit specialist override
.\.venv\Scripts\python -m jarvis.main chat "Deduce the syllogism validity" --specialist openai

# PDF document generation
.\.venv\Scripts\python -m jarvis.main chat "Generate a PDF report titled 'Q1_Summary.pdf' with project highlights"
```

---

## 🧪 Automated Testing & Acceptance Verification

### Run the Full Pytest Suite (73 Tests)
```powershell
.\.venv\Scripts\python -m pytest tests -v
```
All 73 tests cover:
- Multi-model registry, provider health audits, and status reporting.
- OpenAI, Anthropic, and Gemini specialist adapters and error handling.
- NVIDIA Nemotron final synthesis workflow.
- ReportLab PDF generation, styling, and `%PDF-` / `%%EOF` disk verification.
- Secure `/api/downloads/{file_name}` path-traversal defenses.
- Persistent SQLite memory, FTS5 BM25 search, supersession, and isolation.
- Consequential safety gates and AST math evaluation.

### Run the Acceptance Demonstration Script (Demos 1 - 6)
```powershell
.\.venv\Scripts\python scripts/demo_aoris.py
```
Demonstrates all 6 required acceptance scenarios:
1. **Demo 1**: Multi-model status check across all 4 providers.
2. **Demo 2**: Reasoning query routed to OpenAI specialist with Nemotron synthesis.
3. **Demo 3**: Coding query routed to Anthropic specialist with Nemotron synthesis.
4. **Demo 4**: Pure-Python ReportLab PDF generation and structural verification.
5. **Demo 5**: Graceful fallback when a specialist provider is unconfigured.
6. **Demo 6**: End-to-end multi-model execution with Nemotron final synthesis.

---

## ☁️ Production Cloud Deployment Guide

A.O.R.I.S. is containerized and cloud-ready for platforms such as Render, Railway, Fly.io, Google Cloud Run, AWS ECS, or any Linux VPS.

### 1. Docker Deployment
```bash
# Build production container image
docker build -t aoris:latest .

# Run with persistent volume and environment variables
docker run -d \
  -p 8000:8000 \
  -v aoris_data:/app/data \
  -e NEBIUS_API_KEY="your_nebius_key" \
  -e OPENAI_API_KEY="your_openai_key" \
  -e ANTHROPIC_API_KEY="your_anthropic_key" \
  -e GOOGLE_API_KEY="your_gemini_key" \
  --name aoris-app \
  aoris:latest
```

### 2. Cloud Health Probes
- **Liveness Probe**: `GET /healthz` -> Returns `{"status": "ok", "service": "A.O.R.I.S."}`.
- **Readiness Probe**: `GET /api/health` -> Validates SQLite database connection and table readiness.
- **Multi-Model Status Probe**: `GET /api/providers/status` -> Returns operational status for all 4 providers.

---

## 🛡️ Security & Safe Execution Boundary

1. **Path Traversal Defense**: All file readers, file writers, and download endpoints (`/api/downloads/{file_name}`) strictly enforce path resolution within approved root directories and reject relative traversal tokens (`..`).
2. **Consequential Action Gates**: File write operations require explicit human confirmation (`approve_task` / `deny_task`).
3. **Pure-Python PDF Generation**: Zero shell invocations or external binaries, eliminating command injection vectors.
4. **Credential Protection**: Strict redaction in logs; no credentials exposed in client-facing payloads.

---

## 👥 Hackathon Team & Credits
- **Team**: APEX LEGENDS
- **Competition**: Nebius × NVIDIA Global AI Hackathon 2026
- **Primary Orchestrator**: NVIDIA Nemotron (via Nebius Token Factory)
