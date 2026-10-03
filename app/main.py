from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import router
from app.browser import BrowserClient
from app.config import Settings
from app.continuity import ContinuityStore
from app.hermes_chat import deliver_proactive_message, run_life_cognition
from app.life_runtime import LifeScheduler, LifeService, LifeStore
from app.logging_config import configure_logging
from app.memory import MemoryClient
from app.models import ModelProvider
from app.services import build_services


class ExternalContinuityBodyLimitMiddleware:
    """Reject oversized Action writes before JSON parsing or allocation grows."""

    def __init__(self, app: Any, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max(1_024, max_bytes)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path")
            not in {"/v1/continuity/events", "/v1/browser/tasks", "/v1/custom/browser/tasks"}
        ):
            await self.app(scope, receive, send)
            return
        for raw_name, raw_value in scope.get("headers", []):
            if raw_name.lower() != b"content-length":
                continue
            try:
                if int(raw_value) > self.max_bytes:
                    await JSONResponse(
                        status_code=413,
                        content={"detail": "Request body too large"},
                    )(scope, receive, send)
                    return
            except ValueError:
                await JSONResponse(
                    status_code=400,
                    content={"detail": "Malformed Content-Length"},
                )(scope, receive, send)
                return

        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            body = message.get("body", b"")
            total += len(body)
            if total > self.max_bytes:
                await JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large"},
                )(scope, receive, send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break

        complete_body = b"".join(chunks)
        delivered = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {
                    "type": "http.request",
                    "body": complete_body,
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)


def create_app(
    settings: Settings | None = None,
    *,
    provider_override: ModelProvider | None = None,
    memory_override: MemoryClient | None = None,
    browser_override: BrowserClient | None = None,
    continuity_override: ContinuityStore | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    configure_logging(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = build_services(
            resolved_settings,
            provider_override=provider_override,
            memory_override=memory_override,
            browser_override=browser_override,
            continuity_override=continuity_override,
        )
        life_store = LifeStore(resolved_settings.database_path)
        life_service = LifeService(
            life_store,
            lambda request: run_life_cognition(request, settings=resolved_settings),
            outbound_delivery=deliver_proactive_message,
        )
        life_service.initialize()
        life_scheduler = LifeScheduler(life_service)
        app.state.life_runtime = life_service
        life_scheduler.start()
        try:
            yield
        finally:
            await life_scheduler.stop()

    app = FastAPI(
        title="Xiaxia Core",
        version=resolved_settings.version,
        debug=False,
        lifespan=lifespan,
    )
    app.add_middleware(
        ExternalContinuityBodyLimitMiddleware,
        max_bytes=resolved_settings.custom_gpt_continuity_request_max_bytes,
    )
    app.include_router(router)

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        request_id = request.headers.get("X-Request-ID", "unavailable")
        logging.getLogger("xiaxia.api").exception(
            "unhandled_request_error",
            extra={"fields": {"request_id": request_id, "path": request.url.path}},
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": {
                    "code": "internal_error",
                    "message": "Internal server error",
                    "request_id": request_id,
                }
            },
        )

    return app


app = create_app()

assert __version__ == "0.1.0"
