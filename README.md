# proxy-vertex-openai

OpenAI-compatible proxy for Vertex AI third-party models (DeepSeek, Kimi, MiniMax, GLM).
Use with Continue, Cline, OpenCode, or any tool that supports OpenAI API format.

```
Continue / Cline / OpenCode
    ↓  (OpenAI format)
proxy  (injects Google auth)
    ↓
Vertex AI → DeepSeek / Kimi / MiniMax / GLM
```

## Install

```bash
pip install -e .
```

## Setup

**1. Create a Service Account key on GCP**

- Go to GCP Console → IAM & Admin → Service Accounts → Create Service Account
- Assign role: **Agent Platform User** (`roles/aiplatform.user`)
- Go to the service account → Keys tab → Add Key → JSON
- Download the file and rename it to `service-account.json`
- Place it in the same directory where you run the proxy

**2. Create a `.env` file**

```bash
cp .env.example .env
```

Edit `.env`:

```
PROJECT_ID=your-gcp-project-id
```

> **How to find your Project ID:**
> - GCP Console: click the project dropdown at the top → Project ID is shown below the project name
> - Or go to Home → Dashboard → Project info card → **Project ID**
> - It is different from the Project Name

## Usage

```bash
proxy-vertex-openai
```

## Configure your AI tool

Set the following in Continue / Cline / OpenCode / any OpenAI-compatible tool:

```
Base URL : http://127.0.0.1:8082/v1
API Key  : vertex-proxy   (any value)
```

## Available models

| Model alias        | Vertex AI model ID                      |
| ------------------ | --------------------------------------- |
| `deepseek-v3`      | `deepseek-ai/deepseek-v3.2-maas`        |
| `deepseek-v3.2`    | `deepseek-ai/deepseek-v3.2-maas`        |
| `kimi-k2`          | `moonshotai/kimi-k2-thinking-maas`      |
| `kimi-k2-thinking` | `moonshotai/kimi-k2-thinking-maas`      |
| `glm-5`            | `zai-org/glm-5-maas`                    |
| `glm5`             | `zai-org/glm-5-maas`                    |

## Optional flags

```
--port       Port               [env: PROXY_PORT]    default: 8082
--host       Bind address       [env: PROXY_HOST]    default: 127.0.0.1
--log-level  debug|info|warning|error                default: warning
```

## Run tests

```bash
pip install pytest httpx
pytest
```
