"""
proxy-vertex-openai CLI

Setup:
  1. Add PROJECT_ID=your-project to .env
  2. Place service-account.json in the same directory
  3. Run: proxy-vertex-openai
"""

import os
from pathlib import Path

import typer
import uvicorn
from dotenv import load_dotenv

SA_KEY_FILENAME = "service-account.json"


def start(
    port: int = typer.Option(
        8082,
        "--port",
        envvar="PROXY_PORT",
        help="Port to listen on. Default: 8082",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        envvar="PROXY_HOST",
        help="Host to bind to. Default: 127.0.0.1",
    ),
    log_level: str = typer.Option(
        "warning",
        "--log-level",
        help="Uvicorn log level: debug | info | warning | error.",
    ),
    region: str = typer.Option(
        "us-central1",
        "--region",
        envvar=["PROXY_REGION", "VERTEX_REGION"],
        help="Vertex AI region for Veo video generation. Default: us-central1",
    ),
):
    """Start the Vertex AI OpenAI-compatible proxy server."""

    load_dotenv()

    project_id = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        typer.echo(typer.style("Missing PROJECT_ID in .env", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1)

    api_key = os.environ.get("PROXY_API_KEY", "")
    if not api_key:
        typer.echo(typer.style("Missing PROXY_API_KEY in .env", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1)

    sa_key_json = os.environ.get("GOOGLE_SA_JSON", "")

    sa_key = Path(SA_KEY_FILENAME)
    if not sa_key_json and not sa_key.is_file():
        typer.echo(
            typer.style(f"Service account key not found: {SA_KEY_FILENAME}. Set GOOGLE_SA_JSON env var for cloud deployments.", fg=typer.colors.RED),
            err=True,
        )
        raise typer.Exit(code=1)

    # On Railway/cloud, PORT env var overrides --port; HOST defaults to 0.0.0.0
    env_port = os.environ.get("PORT")
    if env_port:
        port = int(env_port)
        host = "0.0.0.0"

    base_url = f"http://{host}:{port}"
    typer.echo(f"Vertex AI Proxy running at {typer.style(base_url, fg=typer.colors.GREEN)}")

    allowed_ips = os.environ.get("ALLOWED_IPS", "")

    from proxy_vertex_openai.app import create_app, parse_allowed_ips

    server_app = create_app(
        project_id=project_id,
        sa_key_path=str(sa_key.resolve()),
        api_key=api_key,
        allowed_ips=allowed_ips,
        region=region,
        sa_key_json=sa_key_json,
    )
    uvicorn.run(server_app, host=host, port=port, log_level=log_level)


def main():
    typer.run(start)


if __name__ == "__main__":
    main()
