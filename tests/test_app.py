"""
Tests for the proxy app — no real Vertex AI calls, all network mocked.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from proxy_vertex_openai.app import MODEL_MAP, _is_allowed, create_app, parse_allowed_ips


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    with patch("proxy_vertex_openai.app.service_account.Credentials.from_service_account_file"):
        fastapi_app = create_app(project_id="test-project", sa_key_path="fake.json", api_key="test-key")
        yield TestClient(fastapi_app, headers={"Authorization": "Bearer test-key"})


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_no_auth(client):
    r = client.get("/health", headers={})
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_wrong_api_key_rejected(client):
    r = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_missing_api_key_rejected(client):
    r = client.get("/v1/models", headers={})
    assert r.status_code == 401


# ── Models ────────────────────────────────────────────────────────────────────

def test_list_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "deepseek-v3" in ids
    assert "kimi-k2" in ids
    assert "glm-5" in ids


def test_model_map_aliases():
    assert "deepseek-v3.2" in MODEL_MAP
    assert "kimi-k2-thinking" in MODEL_MAP
    assert "glm5" in MODEL_MAP


# ── IP whitelist ──────────────────────────────────────────────────────────────

def test_parse_allowed_ips_single():
    assert len(parse_allowed_ips("1.2.3.4")) == 1


def test_parse_allowed_ips_cidr():
    assert len(parse_allowed_ips("10.0.0.0/8,192.168.1.0/24")) == 2


def test_parse_allowed_ips_empty():
    assert parse_allowed_ips("") == []


def test_parse_allowed_ips_invalid_skipped():
    assert len(parse_allowed_ips("not-an-ip,1.2.3.4")) == 1


def test_is_allowed_logic():
    nets = parse_allowed_ips("1.2.3.4,10.0.0.0/8")
    assert _is_allowed("1.2.3.4", nets) is True
    assert _is_allowed("10.5.5.5", nets) is True
    assert _is_allowed("8.8.8.8", nets) is False
    assert _is_allowed("192.168.1.1", nets) is False


def test_blocked_ip_returns_403():
    with patch("proxy_vertex_openai.app.service_account.Credentials.from_service_account_file"):
        fastapi_app = create_app(
            project_id="test-project",
            sa_key_path="fake.json",
            api_key="test-key",
            allowed_ips="1.2.3.4",  # TestClient connects as 127.0.0.1 → blocked
        )
        c = TestClient(fastapi_app, headers={"Authorization": "Bearer test-key"}, raise_server_exceptions=False)
        r = c.get("/v1/models")
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "ip_not_allowed"


def test_no_whitelist_allows_all():
    with patch("proxy_vertex_openai.app.service_account.Credentials.from_service_account_file"):
        fastapi_app = create_app(
            project_id="test-project",
            sa_key_path="fake.json",
            api_key="test-key",
            allowed_ips="",
        )
        c = TestClient(fastapi_app, headers={"Authorization": "Bearer test-key"})
        r = c.get("/v1/models")
        assert r.status_code == 200
