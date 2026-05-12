# proxy-vertex-openai

FastAPI proxy that exposes Vertex AI third-party models (DeepSeek, Kimi, GLM) through a standard **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/models`). Lets any tool that supports OpenAI format — Continue, Cline, OpenCode, Cursor, etc. — use Vertex AI models without code changes.

## Architecture

```
AI tool (Continue / Cline / OpenCode / Cursor)
    ↓  standard OpenAI API format
FastAPI proxy  ← injects Google OAuth2 Bearer token
    ↓  Vertex AI MaaS REST API
Vertex AI → DeepSeek / Kimi / GLM / ...
```

Auth is handled via a GCP service account key (`service-account.json`). The `google-auth` library manages token refresh automatically.

## Project structure

```
src/proxy_vertex_openai/
  app.py   — FastAPI app factory (create_app), MODEL_MAP, route handlers
  cli.py   — Typer CLI, reads .env, launches uvicorn
tests/
  test_app.py    — unit tests, all network calls mocked (no real Vertex AI)
pyproject.toml   — package metadata, entry point: proxy-vertex-openai
.env.example     — required env var template
```

## Setup

```bash
pip install -e .
cp .env.example .env        # edit: PROJECT_ID=your-gcp-project-id
# place service-account.json in the working directory
```

Service account requires role: `roles/aiplatform.user` (Agent Platform User).

## Running

```bash
proxy-vertex-openai                    # binds 127.0.0.1:8082
proxy-vertex-openai --port 9000
proxy-vertex-openai --region us-east5
proxy-vertex-openai --log-level info
```

Configure AI tool:
- **Base URL**: `http://127.0.0.1:8082/v1`
- **API Key**: any value (e.g. `vertex-proxy`)

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | OpenAI chat completions (streaming supported) |

## Adding models

Edit `MODEL_MAP` in `app.py`. Keys are the alias the client sends; values are Vertex AI model IDs:

```python
MODEL_MAP = {
    "deepseek-v3": "deepseek-ai/deepseek-v3.2-maas",
    ...
}
```

## Testing

```bash
pip install pytest httpx
pytest
```

## Sensitive files (never commit)

| File | Covered by .gitignore |
|---|---|
| `.env` | `.env` pattern |
| `service-account.json` | `*.json` pattern |
| `.claude/settings.local.json` | `*.json` + explicit rule |

The `*.json` rule is intentionally broad — there are no JSON config files to track in this project.
