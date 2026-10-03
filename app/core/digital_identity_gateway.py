from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, replace
from typing import Any, Literal

from app.browser import BrowserClient, BrowserClientError
from app.continuity import contains_sensitive_text
from app.core.runtime import ConversationRuntime, RuntimeResult
from app.execution import (
    ChatStatus,
    ExecutionLedger,
    ExecutionRequestClosedError,
    ExecutionRequestSnapshot,
    VerificationState,
)

logger = logging.getLogger("xiaxia.digital_identity_gateway")

GatewaySource = Literal["app", "custom_gpt"]

_MANDATORY_HUMAN_TEXT = re.compile(
    r"captcha|hcaptcha|recaptcha|two.?factor|2fa|one.?time.?code|verification.?code|"
    r"verify.{0,20}(?:human|identity)|验证码|人机验证|身份验证",
    re.IGNORECASE,
)
_LOGIN_TASK = re.compile(
    r"login|log in|sign in|account|current.?identity|digital.?identity|登录|登陆|账号",
    re.IGNORECASE,
)
_LOGIN_PAGE_TEXT = re.compile(r"login|log in|sign in|登录|登陆", re.IGNORECASE)
_EXPLICIT_TAKEOVER_REQUEST = re.compile(
    r"human\s+takeover|发起.{0,12}接管|让我接管|交给我接管|"
    r"\bhand\s+(?:browser\s+)?control\s+to\s+me\b",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_AUTH_ERROR_CODES = {
    "authentication_required",
    "captcha_required",
    "human_verification_required",
    "sensitive_field_blocked",
}


@dataclass(frozen=True, slots=True)
class _LedgerOperation:
    ok: bool
    value: object = None
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class GatewayBrowserOutcome:
    browser_result: Any
    verification_state: VerificationState | None
    evidence_status: str
    request_id: str

    @property
    def claim_verified(self) -> bool:
        return self.evidence_status == "available" and self.verification_state in {
            VerificationState.APPLIED_VERIFIED,
            VerificationState.CAPTURED_NOT_INTERPRETED,
        }


class DigitalIdentityGateway:
    """High-level façade for the one Xiaxia browser identity.

    It deliberately accepts no raw browser action arrays and never forwards takeover
    input to the model, Core prompt, Continuity, or Memory.
    """

    def __init__(
        self,
        *,
        runtime: ConversationRuntime,
        browser: BrowserClient,
        execution_ledger: ExecutionLedger,
        enabled: bool,
        public_base_url: str | None,
        takeover_ttl_seconds: int,
    ) -> None:
        self.runtime = runtime
        self.browser = browser
        self.execution_ledger = execution_ledger
        self.enabled = enabled
        self.public_base_url = public_base_url
        self.takeover_ttl_seconds = takeover_ttl_seconds

    async def _ledger_call(self, operation: str, function, *args, **kwargs):
        try:
            return _LedgerOperation(
                True, await asyncio.to_thread(function, *args, **kwargs)
            )
        except Exception as exc:
            logger.exception(
                "execution_ledger_operation_failed",
                extra={
                    "fields": {
                        "operation": operation,
                        "request_id": args[0] if args else None,
                    }
                },
            )
            return _LedgerOperation(False, error=exc)

    async def _browser_execute(
        self, request_id: str, action: str, arguments: dict[str, Any]
    ) -> GatewayBrowserOutcome:
        begin_operation = asyncio.create_task(
            self._ledger_call(
                "begin_attempt",
                self.execution_ledger.begin_attempt,
                request_id,
                action,
                arguments,
            )
        )
        try:
            begun = await asyncio.shield(begin_operation)
        except asyncio.CancelledError:
            begun = await begin_operation
            if begun.ok and isinstance(begun.value, str):
                await self._ledger_call(
                    "complete_attempt",
                    self.execution_ledger.complete_attempt,
                    begun.value,
                    action=action,
                    error_code="browser_action_cancelled",
                )
            raise
        if isinstance(begun.error, ExecutionRequestClosedError):
            raise BrowserClientError("execution_request_finalized")
        attempt_id = begun.value if begun.ok and isinstance(begun.value, str) else None
        raw_result: Any = None
        try:
            raw_result = await self.browser.execute(action, arguments)
        except BrowserClientError as exc:
            if attempt_id is not None:
                await self._ledger_call(
                    "complete_attempt",
                    self.execution_ledger.complete_attempt,
                    attempt_id,
                    action=action,
                    raw_result=raw_result,
                    error_code=exc.code,
                )
            raise
        except asyncio.CancelledError:
            if attempt_id is not None:
                completion = asyncio.create_task(
                    self._ledger_call(
                        "complete_attempt",
                        self.execution_ledger.complete_attempt,
                        attempt_id,
                        action=action,
                        raw_result=raw_result,
                        error_code="browser_action_cancelled",
                    )
                )
                try:
                    await asyncio.shield(completion)
                except asyncio.CancelledError:
                    await completion
            raise
        verification = None
        completed = _LedgerOperation(False)
        if attempt_id is not None:
            completed = await self._ledger_call(
                "complete_attempt",
                self.execution_ledger.complete_attempt,
                attempt_id,
                action=action,
                raw_result=raw_result,
            )
            verification = (
                completed.value
                if isinstance(completed.value, VerificationState)
                else None
            )
        return GatewayBrowserOutcome(
            browser_result=raw_result,
            verification_state=verification,
            evidence_status=(
                "available" if begun.ok and completed.ok else "unavailable"
            ),
            request_id=request_id,
        )

    async def _standalone_browser_execute(
        self,
        *,
        source: GatewaySource,
        action: str,
        arguments: dict[str, Any],
    ) -> GatewayBrowserOutcome:
        request_id = str(uuid.uuid4())
        begun = await self._ledger_call(
            "begin_request",
            self.execution_ledger.begin_request,
            request_id,
            f"{source}:takeover",
            task_kind="takeover_control",
        )
        try:
            outcome = await self._browser_execute(request_id, action, arguments)
        finally:
            finalized = await self._ledger_call(
                "finalize_request",
                self.execution_ledger.finalize_request,
                request_id,
                chat_status=ChatStatus.COMPLETED,
            )
        if not begun.ok or not finalized.ok:
            outcome = replace(outcome, evidence_status="unavailable")
        return outcome

    @property
    def configured(self) -> bool:
        return bool(
            self.enabled
            and self.browser.configured
            and self.public_base_url
            and self.public_base_url.startswith("https://")
        )

    @staticmethod
    def _page_requires_human(page: Any, task: str) -> bool:
        if not isinstance(page, dict):
            return False
        semantic = page.get("semantic")
        if not isinstance(semantic, dict):
            return False
        inputs = semantic.get("inputs")
        has_password_input = False
        if isinstance(inputs, list):
            for item in inputs:
                if not isinstance(item, dict):
                    continue
                input_type = str(item.get("type") or "").casefold()
                autocomplete = str(item.get("autocomplete") or "").casefold()
                if autocomplete == "one-time-code":
                    return True
                if input_type == "password" or autocomplete in {
                    "current-password",
                    "new-password",
                }:
                    has_password_input = True
        safe_page_markers = " ".join(
            str(semantic.get(key) or "")[:2_000]
            for key in ("title", "headings", "buttons", "visibleText")
        )
        login_page_markers = " ".join(
            str(semantic.get(key) or "")[:2_000]
            for key in ("title", "headings")
        )
        task_without_urls = _URL.sub("", task)
        return bool(
            _MANDATORY_HUMAN_TEXT.search(safe_page_markers)
            or (
                _LOGIN_TASK.search(task_without_urls)
                and (
                    has_password_input
                    or _LOGIN_PAGE_TEXT.search(login_page_markers)
                )
            )
        )

    @staticmethod
    def _continuation_task(
        task: str, snapshot: ExecutionRequestSnapshot | None
    ) -> str:
        """Project only unfinished required steps into the post-takeover request."""

        if snapshot is None:
            return task
        final_attempts = {
            str(item.get("action_id")): item for item in snapshot.final_attempts
        }
        verified_states = {
            VerificationState.APPLIED_VERIFIED.value,
            VerificationState.CAPTURED_NOT_INTERPRETED.value,
        }
        remaining = [
            item
            for item in snapshot.subgoals
            if item.get("required") is True
            and str(final_attempts.get(str(item.get("action_id")), {}).get("verification_state"))
            not in verified_states
        ]
        if not remaining:
            return "在当前页面读取并确认用户完成验证后的状态。"

        clauses: list[str] = []
        for item in remaining:
            original = item.get("original_required_subgoal")
            position = (
                int(original.get("position") or 0)
                if isinstance(original, dict)
                else 0
            )
            left = max(
                task.rfind(mark, 0, max(0, position))
                for mark in ("，", "。", "；", ",", ";", "\n")
            )
            right_candidates = [
                found
                for mark in ("，", "。", "；", ",", ";", "\n")
                if (found := task.find(mark, max(0, position) + 1)) >= 0
            ]
            right = min(right_candidates) if right_candidates else len(task)
            clause = task[left + 1 : right].strip()
            if clause and clause not in clauses:
                clauses.append(clause)
        if not clauses:
            return "在当前页面继续执行原请求中尚未完成的步骤。"
        has_remaining_navigation = any(
            item.get("normalized_action") in {"navigate", "new_tab"}
            for item in remaining
        )
        prefix = (
            "继续执行以下未完成步骤："
            if has_remaining_navigation
            else "在当前页面继续，不要重新导航："
        )
        return prefix + "，然后".join(clauses)

    @staticmethod
    def _task_tab_id(snapshot: ExecutionRequestSnapshot | None) -> str | None:
        if snapshot is None:
            return None
        for attempt in reversed(snapshot.final_attempts):
            if attempt.get("normalized_action") == "status":
                continue
            tab_id = str(attempt.get("tab_id") or "").strip()
            if tab_id:
                return tab_id
        return None

    async def execute(
        self,
        *,
        task: str,
        source: GatewaySource,
        conversation_id: str | None,
        allow_human_takeover: bool,
    ) -> dict[str, Any]:
        if not self.configured:
            raise BrowserClientError("digital_identity_gateway_unavailable", retryable=True)
        if contains_sensitive_text(task):
            raise BrowserClientError("sensitive_browser_task_rejected")

        request_id = str(uuid.uuid4())
        result = await self.runtime.chat(
            message=task,
            conversation_id=conversation_id,
            request_id=request_id,
            continuity_source_override=source,
        )
        if not result.browser_invoked:
            return {
                "status": "rejected",
                "code": "browser_task_not_actionable",
                "requestId": result.request_id,
                "message": "Submit one high-level browse, interact, or current-identity task.",
            }

        takeover = None
        if allow_human_takeover:
            takeover = await self.maybe_create_takeover(
                task=task, source=source, result=result
            )
        if takeover is not None:
            return {
                "status": (
                    "human_takeover_required"
                    if takeover.get("claimStatus") == "verified"
                    else "human_takeover_unverifiable"
                ),
                "code": (
                    "human_takeover_required"
                    if takeover.get("claimStatus") == "verified"
                    else "execution_verification_unavailable"
                ),
                "requestId": result.request_id,
                "conversationId": result.conversation_id,
                "takeover": {
                    key: takeover.get(key)
                    for key in (
                        "sessionId",
                        "takeoverUrl",
                        "expiresAt",
                        "authenticationState",
                    )
                },
                "evidenceStatus": takeover.get("evidenceStatus"),
                "claimStatus": takeover.get("claimStatus"),
                "evidenceRequestId": takeover.get("evidenceRequestId"),
                "instruction": "Open the temporary link. Never send credentials to Xiaxia or Custom GPT.",
            }

        return {
            "status": result.task_status,
            "code": result.browser_error_code,
            "requestId": result.request_id,
            "conversationId": result.conversation_id,
            "reply": result.reply,
            "evidence": {
                "browserInvoked": result.browser_invoked,
                "browserGeneration": result.browser_generation,
                "toolCalls": result.browser_tool_calls,
                "pageUrl": (
                    result.browser_page_url
                    if result.task_status == "completed"
                    else None
                ),
                "pageTitle": (
                    result.browser_page_title
                    if result.task_status == "completed"
                    else None
                ),
                "resultTruncated": result.browser_result_truncated,
                "evidenceStatus": result.execution_evidence_status,
            },
        }

    async def maybe_create_takeover(
        self, *, task: str, source: GatewaySource, result: RuntimeResult
    ) -> dict[str, Any] | None:
        if not self.configured or contains_sensitive_text(task):
            return None
        snapshot_function = getattr(self.execution_ledger, "snapshot", None)
        snapshot = None
        if snapshot_function is not None:
            snapshot_result = await self._ledger_call(
                "snapshot_for_takeover_continuation",
                snapshot_function,
                result.request_id,
            )
            if isinstance(snapshot_result.value, ExecutionRequestSnapshot):
                snapshot = snapshot_result.value
        task_tab_id = self._task_tab_id(snapshot)
        explicit_takeover = bool(_EXPLICIT_TAKEOVER_REQUEST.search(task))
        if result.browser_error_code in _AUTH_ERROR_CODES:
            requires_human = True
        elif not result.browser_invoked and not explicit_takeover:
            return None
        else:
            detection = await self._standalone_browser_execute(
                source=source,
                action="read_page",
                arguments={
                    "format": "semantic",
                    "maxChars": 8_000,
                    **({"tabId": task_tab_id} if task_tab_id else {}),
                },
            )
            requires_human = detection.claim_verified and (
                explicit_takeover
                or self._page_requires_human(detection.browser_result, task)
            )
        if not requires_human:
            return None
        resume_task = self._continuation_task(task, snapshot)
        assert self.public_base_url is not None
        created = await self._standalone_browser_execute(
            source=source,
            action="takeover_create",
            arguments={
                "publicBaseUrl": self.public_base_url,
                "resumeTask": resume_task,
                "source": source,
                "ttlSeconds": self.takeover_ttl_seconds,
                "originRequestId": result.request_id,
                **({"tabId": task_tab_id} if task_tab_id else {}),
            },
        )
        data = (
            dict(created.browser_result)
            if isinstance(created.browser_result, dict)
            else {"browserResult": created.browser_result}
        )
        data["evidenceStatus"] = created.evidence_status
        data["evidenceRequestId"] = created.request_id
        data["claimStatus"] = (
            "verified" if created.claim_verified else "unverifiable"
        )
        return data

    async def resume(
        self, *, session_id: str, source: GatewaySource, conversation_id: str | None
    ) -> dict[str, Any]:
        status_outcome = await self._standalone_browser_execute(
            source=source,
            action="takeover_status",
            arguments={"sessionId": session_id},
        )
        status = status_outcome.browser_result
        if not isinstance(status, dict):
            raise BrowserClientError("malformed_browser_response")
        if status.get("source") != source:
            raise BrowserClientError("takeover_source_mismatch")
        if status.get("state") != "ended":
            return {
                "status": "human_takeover_pending",
                "code": "human_takeover_pending",
                "takeover": {
                    key: status.get(key)
                    for key in ("sessionId", "state", "expiresAt", "authenticationState")
                },
            }
        if status.get("authenticationState") != "authenticated":
            return {
                "status": "failed",
                "code": "human_takeover_not_authenticated",
                "takeover": {
                    key: status.get(key)
                    for key in ("sessionId", "state", "terminalReason", "authenticationState")
                },
            }
        task = status.get("resumeTask")
        if status.get("resumeConsumed") is True:
            return {
                "status": "not_executed",
                "code": "takeover_resume_already_consumed",
                "takeover": {
                    key: status.get(key)
                    for key in ("sessionId", "state", "terminalReason", "authenticationState")
                },
            }
        consumed = await self._standalone_browser_execute(
            source=source,
            action="takeover_consume",
            arguments={"sessionId": session_id},
        )
        consumed_status = consumed.browser_result
        if not isinstance(consumed_status, dict):
            raise BrowserClientError("malformed_browser_response")
        task = consumed_status.get("resumeTask")
        if not isinstance(task, str) or not task or contains_sensitive_text(task):
            raise BrowserClientError("takeover_resume_task_unavailable")
        return await self.execute(
            task=task,
            source=source,
            conversation_id=conversation_id,
            allow_human_takeover=True,
        )

    async def status(self, session_id: str, source: GatewaySource) -> dict[str, Any]:
        outcome = await self._standalone_browser_execute(
            source=source,
            action="takeover_status",
            arguments={"sessionId": session_id},
        )
        result = outcome.browser_result
        if not isinstance(result, dict):
            raise BrowserClientError("malformed_browser_response")
        result = dict(result)
        if result.get("source") != source:
            raise BrowserClientError("takeover_source_mismatch")
        result.pop("resumeTask", None)
        if not outcome.claim_verified:
            result["evidenceStatus"] = outcome.evidence_status
            result["claimStatus"] = "unverifiable"
        return result

    async def revoke(self, session_id: str, source: GatewaySource) -> dict[str, Any]:
        current = await self.status(session_id, source)
        if current.get("state") == "ended":
            return current
        outcome = await self._standalone_browser_execute(
            source=source,
            action="takeover_revoke",
            arguments={"sessionId": session_id},
        )
        result = outcome.browser_result
        if not isinstance(result, dict):
            raise BrowserClientError("malformed_browser_response")
        result = dict(result)
        result.pop("resumeTask", None)
        if not outcome.claim_verified:
            result["evidenceStatus"] = outcome.evidence_status
            result["claimStatus"] = "unverifiable"
        return result
