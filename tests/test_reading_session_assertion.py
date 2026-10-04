import asyncio
import hashlib
from datetime import UTC, datetime
import sys
from types import SimpleNamespace
import types
from uuid import UUID

import pytest
from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from itsdangerous import URLSafeTimedSerializer

if "app.hermes_chat" not in sys.modules:
    # This isolated API test does not exercise the Hermes runtime itself.
    hermes_stub = types.ModuleType("app.hermes_chat")
    hermes_stub.chat = None
    hermes_stub.list_proactive_messages = None
    hermes_stub.register_shared_conversation = None
    sys.modules["app.hermes_chat"] = hermes_stub

from app.api.routes import (
    READING_ASSERTION_AUDIENCE,
    READING_ASSERTION_ROLE,
    READING_ASSERTION_SALT,
    create_reading_session_assertion,
    router,
)
from app.api.schemas import ChatResponse
from fastapi import Response


SECRET = "test-reading-bridge-secret-material-32-bytes-min"


def _request(core_token="core-test", secret=SECRET):
    app = FastAPI()
    app.state.services = SimpleNamespace(
        settings=SimpleNamespace(
            core_api_token=core_token,
            reading_assertion_secret=secret,
        )
    )
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/reading/session",
        "raw_path": b"/v1/reading/session",
        "query_string": b"",
        "headers": [],
        "client": ("test", 1),
        "server": ("test", 443),
        "app": app,
    }
    return Request(scope)


def test_unauthenticated_assertion_request_is_rejected():
    with pytest.raises(HTTPException) as error:
        asyncio.run(create_reading_session_assertion(_request(), Response(), None))
    assert error.value.status_code == 401


def test_valid_core_identity_receives_short_lived_reading_human_assertion():
    response = Response()
    result = asyncio.run(
        create_reading_session_assertion(
            _request(), response, "Bearer core-test"
        )
    )
    claims = URLSafeTimedSerializer(
        SECRET,
        salt=READING_ASSERTION_SALT,
        signer_kwargs={"digest_method": hashlib.sha256},
    ).loads(result["assertion"], max_age=60)

    assert result["expires_in"] == 60
    assert claims["aud"] == READING_ASSERTION_AUDIENCE
    assert claims["role"] == READING_ASSERTION_ROLE
    assert claims["sub"] == "human-owner"
    assert claims["exp"] - claims["iat"] == 60
    assert abs(claims["iat"] - int(datetime.now(UTC).timestamp())) <= 2
    UUID(claims["jti"])
    assert response.headers["Cache-Control"] == "no-store"


def test_bridge_requires_separate_configured_secret():
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            create_reading_session_assertion(
                _request(secret=None), Response(), "Bearer core-test"
            )
        )
    assert error.value.status_code == 503


def test_core_route_surface_contains_no_reading_business_endpoints():
    paths = {route.path for route in router.routes}
    assert "/v1/reading/session" in paths
    assert "/api/books" not in paths
    assert "/mcp" not in paths
    assert "/v1/chat" in paths
    chat_route = next(route for route in router.routes if route.path == "/v1/chat")
    assert chat_route.response_model is ChatResponse
