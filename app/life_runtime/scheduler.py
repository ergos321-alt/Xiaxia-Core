from __future__ import annotations

import asyncio
import logging

from .service import LifeService

logger = logging.getLogger("xiaxia.life_runtime")


class LifeScheduler:
    """One low-cost scheduler in the existing API process."""

    def __init__(self, service: LifeService, *, max_sleep_seconds: float = 60.0) -> None:
        self.service = service
        self.max_sleep_seconds = max(0.05, max_sleep_seconds)
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("LifeScheduler is already running")
        self._task = asyncio.create_task(self._run(), name="xiaxia-life-runtime")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        state = self.service.store.get_state()
        if (
            state.next_meaningful_wake_at is not None
            and state.next_meaningful_wake_at <= self.service.clock.now()
        ):
            await self._safe_tick("restart_due")
        while True:
            state = self.service.store.get_state()
            now = self.service.clock.now()
            deadline = state.next_meaningful_wake_at
            delay = self.max_sleep_seconds if deadline is None else max(
                0.0, (deadline - now).total_seconds()
            )
            await asyncio.sleep(min(delay, self.max_sleep_seconds))
            current = self.service.store.get_state().next_meaningful_wake_at
            if current is not None and self.service.clock.now() >= current:
                await self._safe_tick("scheduled")

    async def _safe_tick(self, reason: str) -> None:
        try:
            result = await self.service.tick(wake_reason=reason)
            logger.info(
                "life_tick_completed",
                extra={"fields": {
                    "wake_reason": reason,
                    "claimed": result.claimed,
                    "decision": result.decision.value if result.decision else None,
                    "appraisal_reason": result.reason,
                    "cognition_called": result.cognition_called,
                }},
            )
        except Exception:
            logger.exception("life_tick_failed", extra={"fields": {"wake_reason": reason}})
