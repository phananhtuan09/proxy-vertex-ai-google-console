import hmac
import ipaddress
import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

VERTEX_BASE = "https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/endpoints/openapi"

# User-facing alias → Vertex AI model ID
MODEL_MAP = {
    "deepseek-v3": "deepseek-ai/deepseek-v3.2-maas",
    "deepseek-v3.2": "deepseek-ai/deepseek-v3.2-maas",
    "kimi-k2": "moonshotai/kimi-k2-thinking-maas",
    "kimi-k2-thinking": "moonshotai/kimi-k2-thinking-maas",
    "glm-5": "zai-org/glm-5-maas",
    "glm5": "zai-org/glm-5-maas",
}

def _unauthorized():
    return JSONResponse(
        status_code=401,
        content={"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}},
    )


def _forbidden():
    return JSONResponse(
        status_code=403,
        content={"error": {"message": "IP not allowed", "type": "invalid_request_error", "code": "ip_not_allowed"}},
    )


def parse_allowed_ips(raw: str) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    networks = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            logger.warning("Invalid IP/CIDR in ALLOWED_IPS, skipping: %s", entry)
    return networks


def _is_allowed(client_ip: str, networks: list) -> bool:
    try:
        addr = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    return any(addr in net for net in networks)


def create_app(project_id: str, sa_key_path: str, api_key: str, allowed_ips: str = "") -> FastAPI:
    app = FastAPI(title="Vertex AI OpenAI Proxy", version="2.0.0")
    ip_networks = parse_allowed_ips(allowed_ips)

    @app.middleware("http")
    async def require_api_key(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        token = auth[len("Bearer "):] if auth.startswith("Bearer ") else ""
        if not hmac.compare_digest(token, api_key):
            return _unauthorized()
        return await call_next(request)

    # Defined last = outermost in Starlette = runs before require_api_key
    @app.middleware("http")
    async def check_allowed_ip(request: Request, call_next):
        if ip_networks:
            client_ip = request.client.host if request.client else ""
            if not _is_allowed(client_ip, ip_networks):
                logger.warning("Blocked request from %s", client_ip)
                return _forbidden()
        return await call_next(request)

    creds = service_account.Credentials.from_service_account_file(
        sa_key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    def access_token() -> str:
        if not creds.valid:
            creds.refresh(GoogleRequest())
        return creds.token

    base_url = VERTEX_BASE.format(project_id=project_id)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {"id": k, "object": "model", "created": 0, "owned_by": "vertex-ai"}
                for k in MODEL_MAP
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()

        model = body.get("model", "")
        body["model"] = MODEL_MAP.get(model, model)

        headers = {
            "Authorization": f"Bearer {access_token()}",
            "Content-Type": "application/json",
        }

        url = f"{base_url}/chat/completions"

        if body.get("stream"):
            async def generate():
                async with httpx.AsyncClient(timeout=120) as client:
                    async with client.stream("POST", url, json=body, headers=headers) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk

            return StreamingResponse(generate(), media_type="text/event-stream", headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            })

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=body, headers=headers)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)

    return app
