import base64
import hmac
import ipaddress
import json
import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

VERTEX_BASE = "https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/endpoints/openapi"
VERTEX_REGIONAL = "https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}"

# User-facing alias → Vertex AI model ID
MODEL_MAP = {
    "deepseek-v3": "deepseek-ai/deepseek-v3.2-maas",
    "deepseek-v3.2": "deepseek-ai/deepseek-v3.2-maas",
    "kimi-k2": "moonshotai/kimi-k2-thinking-maas",
    "kimi-k2-thinking": "moonshotai/kimi-k2-thinking-maas",
    "glm-5": "zai-org/glm-5-maas",
    "glm5": "zai-org/glm-5-maas",
}

# Veo text-to-video models: user alias → Vertex AI model ID
VEO_MODEL_MAP = {
    "veo-3.1": "veo-3.1-generate-001",
    "veo-3.1-generate-001": "veo-3.1-generate-001",
    "veo-3.1-fast": "veo-3.1-fast-generate-001",
    "veo-3.1-fast-generate-001": "veo-3.1-fast-generate-001",
    "veo-3.1-lite": "veo-3.1-lite-generate-001",
    "veo-3.1-lite-generate-001": "veo-3.1-lite-generate-001",
    "veo-3": "veo-3.0-generate-001",
    "veo-3.0-generate-001": "veo-3.0-generate-001",
    "veo-3-fast": "veo-3.0-fast-generate-001",
    "veo-3.0-fast-generate-001": "veo-3.0-fast-generate-001",
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


def create_app(project_id: str, sa_key_path: str, api_key: str, allowed_ips: str = "", region: str = "us-central1", sa_key_json: str = "") -> FastAPI:
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

    if sa_key_json:
        try:
            info = json.loads(base64.b64decode(sa_key_json))
        except Exception:
            info = json.loads(sa_key_json)
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    else:
        creds = service_account.Credentials.from_service_account_file(
            sa_key_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

    def access_token() -> str:
        if not creds.valid:
            creds.refresh(GoogleRequest())
        return creds.token

    base_url = VERTEX_BASE.format(project_id=project_id)
    veo_base = VERTEX_REGIONAL.format(region=region, project_id=project_id)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        all_models = list(MODEL_MAP.keys()) + list(VEO_MODEL_MAP.keys())
        return {
            "object": "list",
            "data": [
                {"id": k, "object": "model", "created": 0, "owned_by": "vertex-ai"}
                for k in all_models
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

    @app.post("/v1/video/generations")
    async def video_generations(request: Request):
        body = await request.json()

        model_alias = body.get("model", "")
        model_id = VEO_MODEL_MAP.get(model_alias, model_alias)

        prompt = body.get("prompt", "")
        parameters = {k: v for k, v in {
            "storageUri": body.get("storage_uri"),
            "sampleCount": body.get("n", body.get("sample_count")),
            "aspectRatio": body.get("aspect_ratio"),
            "durationSeconds": body.get("duration_seconds"),
            "generateAudio": body.get("generate_audio"),
            "negativePrompt": body.get("negative_prompt"),
            "resolution": body.get("resolution"),
            "personGeneration": body.get("person_generation"),
            "compressionQuality": body.get("compression_quality"),
            "enhancePrompt": body.get("enhance_prompt"),
            "resizeMode": body.get("resize_mode"),
            "seed": body.get("seed"),
        }.items() if v is not None}

        headers = {
            "Authorization": f"Bearer {access_token()}",
            "Content-Type": "application/json",
        }

        generate_url = f"{veo_base}/publishers/google/models/{model_id}:predictLongRunning"
        generate_body = {
            "instances": [{"prompt": prompt}],
            "parameters": parameters,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(generate_url, json=generate_body, headers=headers)
            if resp.status_code != 200:
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            operation_name = resp.json().get("name", "")

        return JSONResponse(content={
            "id": operation_name,
            "object": "video.generation",
            "model": model_alias,
            "status": "processing",
            "operation_name": operation_name,
        })

    @app.get("/v1/video/generations/{operation_id:path}")
    async def get_video_generation(operation_id: str, request: Request):
        # operation_id may be the full operation name or just the last segment
        # Reconstruct full name if needed
        if not operation_id.startswith("projects/"):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "operation_id must be the full operation name returned by POST /v1/video/generations", "type": "invalid_request_error"}},
            )

        # Extract model_id from operation name:
        # projects/{proj}/locations/{loc}/publishers/google/models/{model}/operations/{op_id}
        parts = operation_id.split("/")
        try:
            model_idx = parts.index("models") + 1
            model_id = parts[model_idx]
        except (ValueError, IndexError):
            return JSONResponse(status_code=400, content={"error": {"message": "Cannot parse model from operation name", "type": "invalid_request_error"}})

        # Derive region from operation name location
        try:
            loc_idx = parts.index("locations") + 1
            op_region = parts[loc_idx]
        except (ValueError, IndexError):
            op_region = region

        poll_base = VERTEX_REGIONAL.format(region=op_region, project_id=project_id)
        poll_url = f"{poll_base}/publishers/google/models/{model_id}:fetchPredictOperation"
        poll_body = {"operationName": operation_id}

        headers = {
            "Authorization": f"Bearer {access_token()}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(poll_url, json=poll_body, headers=headers)
            if resp.status_code != 200:
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            data = resp.json()

        done = data.get("done", False)
        if not done:
            return JSONResponse(content={
                "id": operation_id,
                "object": "video.generation",
                "status": "processing",
                "operation_name": operation_id,
            })

        if data.get("error"):
            return JSONResponse(content={
                "id": operation_id,
                "object": "video.generation",
                "status": "failed",
                "operation_name": operation_id,
                "error": data["error"],
            })

        # Extract generated videos from response
        # Vertex AI actual format:
        # response.videos[].bytesBase64Encoded + mimeType
        # or response.generateVideoResponse.generatedSamples[].video.uri
        response_payload = data.get("response", {})
        generated_videos = (
            response_payload.get("videos")
            or response_payload.get("generateVideoResponse", {}).get("generatedSamples")
            or response_payload.get("generatedSamples")
            or response_payload.get("generatedVideos")
            or []
        )
        videos_out = []
        for v in generated_videos:
            video_obj = v.get("video", v)
            videos_out.append({
                "uri": video_obj.get("uri") or video_obj.get("gcsUri"),
                "encoding": video_obj.get("encoding") or video_obj.get("mimeType"),
                "video_bytes": (
                    video_obj.get("bytesBase64Encoded")
                    or video_obj.get("videoBytes")
                    or video_obj.get("video_bytes")
                ),
                "mime_type": video_obj.get("mimeType", "video/mp4"),
            })

        filtered_count = (
            response_payload.get("raiMediaFilteredCount")
            or response_payload.get("generateVideoResponse", {}).get("raiMediaFilteredCount")
            or 0
        )
        filtered_reasons = (
            response_payload.get("raiMediaFilteredReasons")
            or response_payload.get("generateVideoResponse", {}).get("raiMediaFilteredReasons")
            or []
        )
        if not videos_out:
            return JSONResponse(content={
                "id": operation_id,
                "object": "video.generation",
                "status": "filtered" if filtered_count else "failed",
                "operation_name": operation_id,
                "data": [],
                "filtered_count": filtered_count,
                "filtered_reasons": filtered_reasons,
                "raw_response": response_payload,
            })

        return JSONResponse(content={
            "id": operation_id,
            "object": "video.generation",
            "status": "succeeded",
            "operation_name": operation_id,
            "data": videos_out,
        })

    return app
