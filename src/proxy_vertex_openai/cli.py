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

    sa_key = Path(SA_KEY_FILENAME)
    if not sa_key.is_file():
        typer.echo(
            typer.style(f"Service account key not found: {SA_KEY_FILENAME}", fg=typer.colors.RED),
            err=True,
        )
        raise typer.Exit(code=1)

    base_url = f"http://{host}:{port}"
    typer.echo(
        f"\n{typer.style('Vertex AI OpenAI Proxy', bold=True)}\n"
        f"  Project  : {typer.style(project_id, fg=typer.colors.CYAN)}\n"
        f"  Listening: {typer.style(base_url, fg=typer.colors.GREEN)}\n"
    )
    typer.echo(
        typer.style("Configure your tool (Continue / Cline / OpenCode):", bold=True) + "\n"
        f"\n  Base URL : {base_url}/v1"
        f"\n  API Key  : {api_key[:4]}{'*' * (len(api_key) - 4)}\n"
        "\n  Available models: deepseek-v3, deepseek-v3.2, kimi-k2, kimi-k2-thinking, glm-5\n"
    )

    allowed_ips = os.environ.get("ALLOWED_IPS", "")

    from proxy_vertex_openai.app import create_app, parse_allowed_ips

    ip_networks = parse_allowed_ips(allowed_ips)
    if ip_networks:
        typer.echo(
            typer.style("  IP whitelist : ", bold=True)
            + typer.style(", ".join(str(n) for n in ip_networks), fg=typer.colors.YELLOW)
            + "\n"
        )
    else:
        typer.echo(typer.style("  IP whitelist : disabled (all IPs allowed)\n", fg=typer.colors.YELLOW))

    server_app = create_app(project_id=project_id, sa_key_path=str(sa_key.resolve()), api_key=api_key, allowed_ips=allowed_ips)
    uvicorn.run(server_app, host=host, port=port, log_level=log_level)


def main():
    typer.run(start)


if __name__ == "__main__":
    main()
