from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Protocol

from .store import LifeActivity, LifeInterior, LifeState, LifeStore


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class GateDecision(str, Enum):
    REST = "REST"
    COGNITION_ELIGIBLE = "COGNITION_ELIGIBLE"


@dataclass(frozen=True, slots=True)
class LifePolicy:
    wake_interval: timedelta = timedelta(hours=6)  # maintenance/retry only; never a Life wake
    quiet_start_hour: int = 22
    quiet_end_hour: int = 7
    local_utc_offset_hours: int = 8
    daily_call_limit: int = 4
    daily_token_limit: int = 20_000
    daily_delivery_limit: int = 3
    minimum_delivery_interval: timedelta = timedelta(hours=2)


@dataclass(frozen=True, slots=True)
class LifeCognitionRequest:
    session_id: str | None
    reason: str
    now: datetime
    activities: tuple[LifeActivity, ...]
    thoughts: tuple[LifeInterior, ...]


@dataclass(frozen=True, slots=True)
class LifeCognitionOutcome:
    session_id: str
    reason: str
    final_response: str | None = None
    noop: bool = True
    thought_id: str | None = None
    thought_content: str | None = None
    thought_status: str | None = None
    revisit_at: datetime | None = None
    usage: dict[str, int] = field(default_factory=dict)


Cognition = Callable[[LifeCognitionRequest], Awaitable[LifeCognitionOutcome]]


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    conversation_id: str
    message_id: str


OutboundDelivery = Callable[[str, str, str], Awaitable[DeliveryResult]]


@dataclass(frozen=True, slots=True)
class TickResult:
    claimed: bool
    decision: GateDecision | None
    reason: str
    cognition_called: bool


class LifeService:
    def __init__(
        self,
        store: LifeStore,
        cognition: Cognition,
        *,
        clock: Clock | None = None,
        policy: LifePolicy | None = None,
        outbound_delivery: OutboundDelivery | None = None,
    ) -> None:
        self.store = store
        self.cognition = cognition
        self.clock = clock or SystemClock()
        self.policy = policy or LifePolicy()
        self.outbound_delivery = outbound_delivery

    def initialize(self) -> None:
        self.store.initialize(now=self.clock.now(), first_wake_after=self.policy.wake_interval)

    def appraise(
        self,
        state: LifeState,
        activities: list[LifeActivity],
        thoughts: list[LifeInterior],
        now: datetime,
    ) -> tuple[GateDecision, str]:
        if activities:
            return GateDecision.COGNITION_ELIGIBLE, "activity_due"
        if state.mode == "dormant" or state.budget_pressure:
            return GateDecision.REST, "budget_or_dormant"
        local_hour = (now + timedelta(hours=self.policy.local_utc_offset_hours)).hour
        if self.policy.quiet_start_hour <= local_hour or local_hour < self.policy.quiet_end_hour:
            return GateDecision.REST, "quiet_hours"
        if thoughts:
            return GateDecision.COGNITION_ELIGIBLE, "thought_revisit_due"
        return GateDecision.REST, "nothing_due"

    async def tick(self, *, wake_reason: str = "scheduled") -> TickResult:
        now = self.clock.now()
        claim = self.store.claim_due_wake(now, reason=wake_reason)
        if claim is None:
            return TickResult(False, None, "not_due_or_already_claimed", False)

        self.store.ledger(
            "WAKE", wake_reason, now=now, receipt_status="completed",
            activity_id=claim.activity_id, thought_id=claim.thought_id,
            details={
                "trigger": claim.trigger_type,
                "due_at": claim.due_at.isoformat(),
                "claimed_at": now.isoformat(),
            },
        )
        if wake_reason == "restart_due":
            self.store.ledger("RECOVERY", "single_restart_appraisal", now=now)
        activities = self.store.due_activities(now)
        thoughts = self.store.due_thoughts(now)
        candidate = next((item for item in thoughts if item.status == "candidate"), None)
        if candidate is not None:
            self.store.ledger(
                "APPRAISE", "outbound_candidate", now=now,
                thought_id=candidate.thought_id,
                details={"decision": "DELIVERY_GATE"},
            )
            delivery_reason = await self._attempt_delivery(candidate, now=now)
            self.store.complete_wake(
                claim.token, now=now, mode="idle",
            )
            return TickResult(True, GateDecision.REST, delivery_reason, False)
        decision, reason = self.appraise(claim.state, activities, thoughts, now)
        self.store.ledger(
            "APPRAISE", reason, now=now,
            activity_id=activities[0].activity_id if activities else None,
            thought_id=thoughts[0].thought_id if thoughts else None,
            details={"decision": decision.value},
        )

        if decision is GateDecision.REST:
            self.store.ledger("REST", reason, now=now)
            self.store.complete_wake(
                claim.token, now=now,
                mode="dormant" if claim.state.budget_pressure else "idle",
            )
            return TickResult(True, decision, reason, False)

        request = LifeCognitionRequest(
            session_id=claim.state.life_session_id,
            reason=reason,
            now=now,
            activities=tuple(activities[:1]),
            thoughts=tuple(thoughts[:1]) if not activities else (),
        )
        started_id = self.store.ledger(
            "COGNITION_STARTED", reason, now=now,
            activity_id=request.activities[0].activity_id if request.activities else None,
            thought_id=request.thoughts[0].thought_id if request.thoughts else None,
            hermes_session_id=request.session_id,
            usage_unknown=True,
            receipt_status="incomplete",
        )
        outcome: LifeCognitionOutcome | None = None
        try:
            outcome = await self.cognition(request)
            completed_at = self.clock.now()
            candidate_id = (
                self._apply_scheduled_result(outcome, request=request, now=completed_at)
                if request.activities
                else self._apply_outcome(outcome, request=request, now=completed_at)
            )
            usage = outcome.usage
            usage_known = "total_tokens" in usage
            self.store.update_receipt(
                started_id, receipt_status="completed", now=completed_at,
                hermes_session_id=outcome.session_id, usage=usage,
                usage_unknown=not usage_known,
                details={"reason": outcome.reason},
            )
            self.store.ledger(
                "COGNITION_COMPLETED", outcome.reason, now=completed_at,
                hermes_session_id=outcome.session_id, usage=usage,
                usage_unknown=not usage_known,
            )
            self.store.set_life_session(outcome.session_id, now=completed_at)
            updated = self.store.record_usage(
                now=completed_at,
                api_calls=usage.get("api_calls", 1),
                total_tokens=usage.get("total_tokens") if usage_known else None,
                call_limit=self.policy.daily_call_limit,
                token_limit=self.policy.daily_token_limit,
            )
            if updated.budget_pressure:
                self.store.ledger("BUDGET_CHANGE", "daily_threshold", now=completed_at,
                                  details={"mode": updated.mode})
            if candidate_id is not None:
                await self._attempt_delivery(
                    self.store.get_thought(candidate_id), now=completed_at
                )
            self.store.complete_wake(
                claim.token, now=completed_at,
                mode="dormant" if updated.budget_pressure else "idle",
            )
            return TickResult(True, decision, reason, True)
        except Exception as exc:
            failed_at = self.clock.now()
            failed_usage = outcome.usage if outcome is not None else {}
            usage_known = outcome is not None and "total_tokens" in failed_usage
            failed_session = outcome.session_id if outcome is not None else request.session_id
            self.store.update_receipt(
                started_id, receipt_status="error", now=failed_at,
                hermes_session_id=failed_session, usage=failed_usage,
                usage_unknown=not usage_known,
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
            self.store.ledger(
                "COGNITION_ERROR", type(exc).__name__, now=failed_at,
                hermes_session_id=failed_session, usage=failed_usage,
                usage_unknown=not usage_known,
                receipt_status="error", details={"error": str(exc)},
            )
            self.store.record_usage(
                now=failed_at, api_calls=failed_usage.get("api_calls", 1),
                total_tokens=failed_usage.get("total_tokens") if usage_known else None,
                call_limit=self.policy.daily_call_limit,
                token_limit=self.policy.daily_token_limit,
            )
            self.store.complete_wake(
                claim.token, now=failed_at,
                mode="dormant",
            )
            raise

    def _apply_outcome(
        self,
        outcome: LifeCognitionOutcome,
        *,
        request: LifeCognitionRequest,
        now: datetime,
    ) -> str | None:
        candidate_id: str | None = None
        allowed_thoughts = {item.thought_id for item in request.thoughts}
        if outcome.thought_id:
            if outcome.thought_id not in allowed_thoughts or not outcome.thought_status:
                raise ValueError("cognition referenced a thought outside its input")
            if outcome.thought_status == "shared":
                raise ValueError("shared status requires successful delivery")
            before = self.store.get_thought(outcome.thought_id)
            after = self.store.transition_thought(
                outcome.thought_id, outcome.thought_status, now=now,
                content=outcome.thought_content,
                revisit_at=(outcome.revisit_at or now)
                if outcome.thought_status == "candidate"
                else outcome.revisit_at,
            )
            self.store.ledger(
                "INTERIOR_TRANSITION", f"{before.status}->{after.status}", now=now,
                thought_id=after.thought_id,
            )
            if after.status == "candidate":
                candidate_id = after.thought_id
                self.store.ledger("OUTBOUND_CANDIDATE", "interior_candidate", now=now,
                                  thought_id=after.thought_id)
        elif outcome.thought_content:
            if outcome.thought_status == "shared":
                raise ValueError("shared status requires successful delivery")
            thought = self.store.add_thought(
                content=outcome.thought_content, now=now,
                status=outcome.thought_status or "private",
                revisit_at=(outcome.revisit_at or now)
                if outcome.thought_status == "candidate"
                else outcome.revisit_at,
                source_activity_id=None,
            )
            self.store.ledger("INTERIOR_TRANSITION", "created", now=now,
                              thought_id=thought.thought_id)
            if thought.status == "candidate":
                candidate_id = thought.thought_id
                self.store.ledger("OUTBOUND_CANDIDATE", "interior_candidate", now=now,
                                  thought_id=thought.thought_id)
        return candidate_id

    def _apply_scheduled_result(
        self,
        outcome: LifeCognitionOutcome,
        *,
        request: LifeCognitionRequest,
        now: datetime,
    ) -> str | None:
        """Finish one claimed event from native Hermes facts, without semantic judging."""
        if len(request.activities) != 1:
            raise ValueError("scheduled cognition requires exactly one claimed activity")
        activity = request.activities[0]
        candidate_id: str | None = None
        if outcome.final_response is not None:
            content = outcome.final_response.strip()
            if not content:
                raise ValueError("completed scheduled cognition returned an empty response")
            thought = self.store.add_thought(
                content=content,
                now=now,
                status="candidate",
                revisit_at=now,
                source_activity_id=activity.activity_id,
            )
            candidate_id = thought.thought_id
            self.store.ledger(
                "INTERIOR_TRANSITION", "created", now=now,
                thought_id=thought.thought_id,
            )
            self.store.ledger(
                "OUTBOUND_CANDIDATE", "hermes_terminal_response", now=now,
                activity_id=activity.activity_id,
                thought_id=thought.thought_id,
            )
        before = self.store.get_activity(activity.activity_id)
        after = self.store.transition_activity(
            activity.activity_id,
            "completed",
            now=now,
            next_wake_at=None,
        )
        self.store.ledger(
            "ACTIVITY_TRANSITION", f"{before.status}->{after.status}", now=now,
            activity_id=after.activity_id,
        )
        return candidate_id

    async def _attempt_delivery(self, thought: LifeInterior, *, now: datetime) -> str:
        state = self.store.get_state()
        facts = self.store.delivery_facts(
            thought.thought_id,
            now=now,
            local_utc_offset_hours=self.policy.local_utc_offset_hours,
        )
        suppression: str | None = None
        if thought.status != "candidate":
            suppression = "not_candidate"
        elif facts["succeeded"]:
            suppression = "already_delivered"
        elif facts["incomplete"]:
            suppression = "delivery_inflight_or_unknown"
        elif facts["daily_count"] >= self.policy.daily_delivery_limit:
            suppression = "daily_limit"
        elif (
            facts["last_succeeded_at"] is not None
            and now - facts["last_succeeded_at"] < self.policy.minimum_delivery_interval
        ):
            suppression = "minimum_interval"
        elif state.shared_conversation_id is None:
            suppression = "no_shared_conversation"
        elif self.outbound_delivery is None:
            suppression = "delivery_boundary_unavailable"

        if suppression is not None:
            self.store.ledger(
                "DELIVERY_SUPPRESSED", suppression, now=now,
                thought_id=thought.thought_id,
            )
            if thought.status == "candidate" and suppression not in {
                "already_delivered", "delivery_inflight_or_unknown", "not_candidate"
            }:
                self.store.transition_thought(
                    thought.thought_id, "candidate", now=now,
                    revisit_at=self._next_delivery_attempt(now, suppression, facts),
                )
            return suppression

        attempt_id = self.store.claim_delivery_attempt(
            thought.thought_id, now=now, reason="delivery_gate_passed"
        )
        if attempt_id is None:
            self.store.ledger(
                "DELIVERY_SUPPRESSED", "duplicate_or_inflight", now=now,
                thought_id=thought.thought_id,
            )
            return "duplicate_or_inflight"

        try:
            assert state.shared_conversation_id is not None
            assert self.outbound_delivery is not None
            result = await self.outbound_delivery(
                state.shared_conversation_id, thought.thought_id, thought.content
            )
        except Exception as exc:
            self.store.update_receipt(
                attempt_id, receipt_status="error", now=now,
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
            self.store.ledger(
                "DELIVERY_FAILED", type(exc).__name__, now=now,
                thought_id=thought.thought_id, receipt_status="error",
                details={"attempt_id": attempt_id, "error": str(exc)},
            )
            self.store.transition_thought(
                thought.thought_id, "candidate", now=now,
                revisit_at=now + self.policy.wake_interval,
            )
            return "delivery_failed"

        self.store.update_receipt(
            attempt_id, receipt_status="completed", now=now,
            details={
                "conversation_id": result.conversation_id,
                "message_id": result.message_id,
            },
        )
        self.store.transition_thought(thought.thought_id, "shared", now=now)
        self.store.ledger(
            "INTERIOR_TRANSITION", "candidate->shared", now=now,
            thought_id=thought.thought_id,
        )
        self.store.ledger(
            "DELIVERY_SUCCEEDED", "shared_conversation_append", now=now,
            thought_id=thought.thought_id,
            receipt_status="completed",
            details={
                "attempt_id": attempt_id,
                "conversation_id": result.conversation_id,
                "message_id": result.message_id,
            },
        )
        return "delivery_succeeded"

    def _next_delivery_attempt(
        self, now: datetime, suppression: str, facts: dict[str, Any]
    ) -> datetime:
        if suppression == "minimum_interval" and facts["last_succeeded_at"] is not None:
            return facts["last_succeeded_at"] + self.policy.minimum_delivery_interval
        if suppression == "daily_limit":
            local = now + timedelta(hours=self.policy.local_utc_offset_hours)
            next_local = (local + timedelta(days=1)).replace(
                hour=self.policy.quiet_end_hour, minute=0, second=0, microsecond=0
            )
            return next_local - timedelta(hours=self.policy.local_utc_offset_hours)
        return now + self.policy.wake_interval
