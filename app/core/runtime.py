from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, replace

from app.continuity import (
    ContinuityCapsule,
    ContinuityError,
    ContinuityEventInput,
    ContinuityStore,
    safe_bounded_summary,
)
from app.core.browser_orchestrator import BrowserOrchestrator
from app.core.context import ContextAssembler
from app.core.native_browser import run_native_browser
from app.execution import (
    ChatStatus,
    ClaimGate,
    ClaimGateResult,
    ExecutionLedger,
    TaskStatus,
)
from app.identity import IdentityBundle
from app.memory import MemoryClient, MemoryItem
from app.models import GenerateResult, ModelProvider, QwenProvider
from app.storage import RuntimeStore

logger = logging.getLogger("xiaxia.runtime")


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    reply: str
    conversation_id: str
    request_id: str
    model: str
    retrieved_memory_count: int
    memory_degraded: bool
    browser_invoked: bool
    browser_degraded: bool
    browser_tool_calls: int
    browser_generation: int | None
    browser_result_truncated: bool
    browser_intent_decision: str
    destination_resolution_attempted: bool
    browser_destination_url: str | None
    browser_page_url: str | None
    browser_page_title: str | None
    browser_error_code: str | None
    continuity_degraded: bool
    continuity_capsule_event_count: int
    continuity_capsule_char_count: int
    continuity_events_written: int
    task_status: str = TaskStatus.NOT_EXECUTED.value
    chat_completed: bool = True
    execution_evidence_status: str = "available"


class ConversationRuntime:
    def __init__(
        self,
        *,
        identity: IdentityBundle,
        assembler: ContextAssembler,
        provider: ModelProvider,
        memory: MemoryClient,
        store: RuntimeStore,
        browser: BrowserOrchestrator,
        browser_native_tool_calling_enabled: bool = False,
        continuity: ContinuityStore,
        continuity_enabled: bool,
        continuity_xiaxia_identity_id: str,
        continuity_user_identity_id: str,
        continuity_source: str,
        continuity_capsule_max_events: int,
        continuity_capsule_max_chars: int,
        continuity_recent_days: int,
        execution_ledger: ExecutionLedger,
        claim_gate: ClaimGate,
    ) -> None:
        self.identity = identity
        self.assembler = assembler
        self.provider = provider
        self.memory = memory
        self.store = store
        self.browser = browser
        self.browser_native_tool_calling_enabled = browser_native_tool_calling_enabled
        self.continuity = continuity
        self.continuity_enabled = continuity_enabled
        self.continuity_xiaxia_identity_id = continuity_xiaxia_identity_id
        self.continuity_user_identity_id = continuity_user_identity_id
        self.continuity_source = continuity_source
        self.continuity_capsule_max_events = continuity_capsule_max_events
        self.continuity_capsule_max_chars = continuity_capsule_max_chars
        self.continuity_recent_days = continuity_recent_days
        self.execution_ledger = execution_ledger
        self.claim_gate = claim_gate

    async def _retrieve_memory(
        self, message: str
    ) -> tuple[list[MemoryItem], list[MemoryItem], bool]:
        if not self.memory.configured:
            return [], [], False

        results = await asyncio.gather(
            self.memory.search_memory(
                message, self.assembler.policy.memory_search_limit
            ),
            self.memory.get_recent_context(self.assembler.policy.memory_recent_limit),
            return_exceptions=True,
        )
        relevant: list[MemoryItem] = []
        recent: list[MemoryItem] = []
        degraded = False
        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                degraded = True
                logger.warning(
                    "memory_retrieval_failed",
                    extra={
                        "fields": {"operation": "search" if index == 0 else "recent"}
                    },
                )
            elif index == 0:
                relevant = result
            else:
                recent = result
        return relevant, recent, degraded

    async def _ledger_call(self, operation: str, function, *args, **kwargs):
        try:
            return True, await asyncio.to_thread(function, *args, **kwargs)
        except Exception:
            logger.exception(
                "execution_ledger_operation_failed",
                extra={
                    "fields": {
                        "operation": operation,
                        "request_id": args[0] if args else None,
                    }
                },
            )
            return False, None

    async def _load_continuity_capsule(
        self, request_id: str
    ) -> tuple[ContinuityCapsule, bool]:
        if not self.continuity_enabled:
            return ContinuityCapsule(), False
        if not self.continuity.configured:
            logger.warning(
                "continuity_capsule_load_failed",
                extra={
                    "fields": {"request_id": request_id, "reason": "not_configured"}
                },
            )
            return ContinuityCapsule(), True
        try:
            capsule = await asyncio.to_thread(
                self.continuity.load_capsule,
                xiaxia_identity_id=self.continuity_xiaxia_identity_id,
                user_identity_id=self.continuity_user_identity_id,
                max_events=self.continuity_capsule_max_events,
                max_chars=self.continuity_capsule_max_chars,
                recent_days=self.continuity_recent_days,
            )
            logger.info(
                "continuity_capsule_loaded",
                extra={
                    "fields": {
                        "request_id": request_id,
                        "capsule_event_count": len(capsule.events),
                        "capsule_char_count": capsule.char_count,
                        "capsule_truncated": capsule.truncated,
                    }
                },
            )
            return capsule, False
        except ContinuityError:
            logger.warning(
                "continuity_capsule_load_failed",
                extra={
                    "fields": {
                        "request_id": request_id,
                        "reason": "storage_unavailable",
                    }
                },
            )
            return ContinuityCapsule(), True

    async def _write_continuity_event(self, event: ContinuityEventInput) -> bool:
        if not self.continuity_enabled or not self.continuity.configured:
            return False
        try:
            stored = await asyncio.to_thread(self.continuity.create_event, event)
            logger.info(
                "continuity_event_written",
                extra={
                    "fields": {
                        "request_id": event.request_id,
                        "event_id": stored.event_id,
                        "source": stored.source,
                        "event_type": stored.event_type,
                    }
                },
            )
            return True
        except ContinuityError:
            logger.warning(
                "continuity_event_write_failed",
                extra={
                    "fields": {
                        "request_id": event.request_id,
                        "source": event.source,
                        "event_type": event.event_type,
                    }
                },
            )
            return False

    @staticmethod
    def _unfinished_topic(message: str) -> str | None:
        normalized = " ".join(message.split())
        match = re.search(
            r"(?:下一步|接下来|待办|还需要|尚未完成)(?:准备|计划|要|是|：|:|\s)*(.+)",
            normalized,
            re.IGNORECASE,
        )
        if not match:
            return None
        topic = match.group(1).strip("。！？.!? ")
        if not topic:
            return None
        return safe_bounded_summary(f"未完成事项：{topic}", limit=500)

    @staticmethod
    def _near_term_plan(message: str) -> str | None:
        """Recognize explicit, time-bound personal intent without model inference."""

        normalized = " ".join(message.split())
        if len(normalized) < 6:
            return None
        has_personal_subject = re.search(
            r"(?:我|我们|咱们)|\b(?:I|we)\b", normalized, re.IGNORECASE
        )
        has_near_term_time = re.search(
            r"(?:今|明|后)(?:天|日|早|晚)"
            r"|(?:这|本|下)(?:周|星期|个月)"
            r"|(?:待|等|过|一)会(?:儿)?|稍后|晚点"
            r"|(?:早上|上午|中午|下午|晚上)"
            r"|(?:下班|放学|起床|吃饭|睡觉)(?:前|后|以后|之前)?"
            r"|\b(?:tonight|tomorrow|later|this\s+(?:week|weekend)|next\s+(?:week|month))\b",
            normalized,
            re.IGNORECASE,
        )
        has_intent = re.search(
            r"(?:想(?:要)?|打算|计划|准备|决定|答应|约好|记得(?:要)?|别忘(?:了)?|到时(?:要)?)"
            r"|\b(?:want\s+to|plan\s+to|intend\s+to|going\s+to|need\s+to|remember\s+to)\b",
            normalized,
            re.IGNORECASE,
        )
        if not (has_personal_subject and has_near_term_time and has_intent):
            return None
        return safe_bounded_summary(f"近期计划：{normalized}", limit=500)

    @staticmethod
    def _conversation_importance(
        message: str, *, browser_invoked: bool, has_unfinished_topic: bool
    ) -> int:
        if browser_invoked or has_unfinished_topic:
            return 6
        if re.search(r"已经|完成|决定|计划|准备|下一步|刚才|今天", message):
            return 6
        return 3

    @staticmethod
    def _cross_entry_handoff(capsule: ContinuityCapsule):
        return next(
            (
                event
                for event in reversed(capsule.events)
                if event.status == "active"
                and isinstance(event.metadata.get("topic_key"), str)
                and event.metadata["topic_key"]
            ),
            None,
        )

    @staticmethod
    def _handoff_intent(message: str) -> str | None:
        if re.search(
            r"(?:接力|刚才|上次|那个|这件|这个).{0,12}(?:已完成|完成了|结束|关闭|解决了)",
            message,
        ):
            return "resolved"
        if re.search(r"(?:继续|接着|刚才|上次|接力)", message):
            return "active"
        return None

    @staticmethod
    def _historical_browser_evidence(
        capsule: ContinuityCapsule,
        conversation_id: str,
    ) -> tuple[dict[str, object], ...]:
        evidence: list[dict[str, object]] = []
        for event in reversed(capsule.events):
            if (
                event.event_type != "browser_event"
                or event.conversation_id != conversation_id
            ):
                continue
            evidence.append(
                {
                    "summary": event.summary,
                    "request_id": event.request_id,
                    "occurred_at": event.occurred_at.isoformat(),
                    "metadata": event.metadata,
                }
            )
            if len(evidence) >= 8:
                break
        return tuple(evidence)

    def _base_event(
        self,
        *,
        source: str,
        event_type: str,
        summary: str,
        importance: int,
        request_id: str,
        conversation_id: str,
        source_event_id: str,
        metadata: dict[str, object] | None = None,
        status: str = "active",
    ) -> ContinuityEventInput:
        return ContinuityEventInput(
            xiaxia_identity_id=self.continuity_xiaxia_identity_id,
            user_identity_id=self.continuity_user_identity_id,
            source=source,  # type: ignore[arg-type]
            event_type=event_type,  # type: ignore[arg-type]
            summary=summary,
            importance=importance,
            status=status,
            metadata=metadata or {},
            request_id=request_id,
            conversation_id=conversation_id,
            source_event_id=source_event_id,
        )

    async def chat(
        self,
        *,
        message: str,
        conversation_id: str | None,
        request_id: str | None = None,
        continuity_source_override: str | None = None,
    ) -> RuntimeResult:
        started = time.perf_counter()
        identifier = await asyncio.to_thread(
            self.store.ensure_conversation, conversation_id
        )
        req_id = request_id or str(uuid.uuid4())
        ledger_ok, _ = await self._ledger_call(
            "begin_request", self.execution_ledger.begin_request, req_id, identifier
        )
        status = "error"
        task_status = TaskStatus.NOT_EXECUTED.value
        retrieved_count = 0
        degraded = False
        browser_invoked = False
        browser_degraded = False
        browser_tool_calls = 0
        browser_generation: int | None = None
        browser_result_truncated = False
        browser_intent_decision = "none"
        destination_resolution_attempted = False
        browser_destination_url: str | None = None
        continuity_degraded = False
        continuity_capsule_event_count = 0
        continuity_capsule_char_count = 0
        continuity_events_written = 0
        execution_ledger_degraded = not ledger_ok
        request_finalized = False

        try:
            history = await asyncio.to_thread(
                self.store.get_messages,
                identifier,
                self.assembler.policy.max_history_messages,
            )
            relevant, recent, degraded = await self._retrieve_memory(message)
            retrieved_count = len(relevant) + len(recent)
            capsule, continuity_degraded = await self._load_continuity_capsule(req_id)
            continuity_capsule_event_count = len(capsule.events)
            continuity_capsule_char_count = capsule.char_count
            model_messages = self.assembler.assemble(
                identity=self.identity,
                relevant_memories=relevant,
                recent_context=recent,
                conversation_history=history,
                current_message=message,
                continuity_capsule=capsule,
            )
            native_result = None
            if (
                self.browser_native_tool_calling_enabled
                and self.browser.enabled
                and isinstance(self.provider, QwenProvider)
            ):
                native_result = await asyncio.wait_for(
                    run_native_browser(
                        self.provider, model_messages, self.browser.tool, req_id
                    ),
                    timeout=self.browser.overall_timeout_seconds,
                )
                browser = native_result.browser
            else:
                browser = await self.browser.augment(
                    message=message,
                    request_id=req_id,
                    model_messages=model_messages,
                    historical_browser_evidence=self._historical_browser_evidence(
                        capsule, identifier
                    ),
                    conversation_id=identifier,
                )
            model_messages = browser.messages
            browser_invoked = browser.invoked
            browser_degraded = browser.degraded
            browser_tool_calls = browser.tool_calls
            browser_generation = browser.generation
            browser_result_truncated = browser.truncated
            browser_intent_decision = browser.intent_decision
            destination_resolution_attempted = browser.destination_resolution_attempted
            browser_destination_url = browser.resolved_url
            execution_ledger_degraded = (
                execution_ledger_degraded or browser.ledger_degraded
            )
            if (
                self.continuity_enabled
                and self.continuity.configured
                and browser.invoked
                and not browser.degraded
                and browser.page_url
                and browser.page_title
            ):
                wrote_browser = await self._write_continuity_event(
                    self._base_event(
                        source="browser",
                        event_type="browser_event",
                        summary=safe_bounded_summary(
                            "浏览器已完成本次请求中的已验证操作"
                            f"（{', '.join(str(item.get('action')) for item in browser.execution_facts)}），"
                            f"最终页面为“{browser.page_title}”（{browser.page_url}）。",
                            limit=500,
                        ),
                        importance=7,
                        request_id=req_id,
                        conversation_id=identifier,
                        source_event_id=f"chat:{req_id}:browser",
                        metadata={
                            "target_url": browser.page_url,
                            "title": safe_bounded_summary(
                                browser.page_title, limit=300
                            ),
                            "browser_generation": browser.generation,
                            "result_truncated": browser.truncated,
                            "execution_facts": list(browser.execution_facts),
                        },
                    )
                )
                if wrote_browser:
                    continuity_events_written += 1
                else:
                    continuity_degraded = True
            ledger_ok, evaluated = await self._ledger_call(
                "evaluate_request", self.execution_ledger.evaluate_request, req_id
            )
            execution_ledger_degraded = execution_ledger_degraded or not ledger_ok
            if isinstance(evaluated, TaskStatus):
                task_status = evaluated.value
            generated_candidate = (
                native_result.generated
                if native_result is not None
                else GenerateResult(text=browser.direct_reply, model="xiaxia-core")
                if browser.direct_reply is not None
                else await self.provider.generate(model_messages)
            )
            ledger_ok, claim_snapshot = await self._ledger_call(
                "snapshot",
                self.execution_ledger.snapshot,
                browser.claim_evidence_request_id or req_id,
            )
            execution_ledger_degraded = execution_ledger_degraded or not ledger_ok
            if (
                claim_snapshot is not None
                and browser.intent_decision != "historical_browser_evidence"
            ):
                claim_snapshot = replace(claim_snapshot, task_status=task_status)
            candidate_text = generated_candidate.text
            if (
                browser.degraded
                and claim_snapshot is not None
                and claim_snapshot.task_status == TaskStatus.PARTIAL.value
            ):
                candidate_text = " ".join(
                    item
                    for item in (
                        browser.direct_reply,
                        self.claim_gate.truthful_summary(claim_snapshot),
                    )
                    if item
                )
            try:
                gated = self.claim_gate.gate(
                    candidate_text,
                    claim_snapshot,
                    evidence_available=not execution_ledger_degraded,
                    execution_context=(
                        browser.invoked
                        or browser.degraded
                        or browser.intent_decision
                        not in {"none", "capability_self_knowledge"}
                    ),
                    historical=(
                        browser.intent_decision == "historical_browser_evidence"
                    ),
                    browser_error_code=browser.error_code,
                )
            except Exception:
                execution_ledger_degraded = True
                logger.exception(
                    "execution_claim_gate_failed",
                    extra={"fields": {"request_id": req_id}},
                )
                gated = ClaimGateResult(
                    text=self.claim_gate.unavailable_summary(browser.error_code),
                    allowed=False,
                    blocked_actions=("__claim_gate_unavailable__",),
                )
            if not gated.allowed:
                logger.warning(
                    "execution_claim_blocked",
                    extra={
                        "fields": {
                            "request_id": req_id,
                            "evidence_request_id": (
                                browser.claim_evidence_request_id or req_id
                            ),
                            "blocked_actions": list(gated.blocked_actions),
                        }
                    },
                )
            generated = GenerateResult(text=gated.text, model=generated_candidate.model)
            await asyncio.to_thread(
                self.store.append_exchange, identifier, message, generated.text
            )
            conversation_summary = safe_bounded_summary(
                f"用户：{safe_bounded_summary(message, limit=200)} "
                f"夏夏答复：{safe_bounded_summary(generated.text, limit=250)}",
                limit=500,
            )
            handoff = (
                self._cross_entry_handoff(capsule)
                if continuity_source_override is None
                and self.continuity_source == "app"
                else None
            )
            handoff_intent = self._handoff_intent(message) if handoff else None
            unfinished = self._unfinished_topic(message)
            unfinished_category = None
            if unfinished is None:
                unfinished = self._near_term_plan(message)
                unfinished_category = "near_term_plan"
            wrote_conversation = await self._write_continuity_event(
                self._base_event(
                    source=continuity_source_override or self.continuity_source,
                    event_type="conversation_event",
                    summary=conversation_summary,
                    importance=self._conversation_importance(
                        message,
                        browser_invoked=browser_invoked,
                        has_unfinished_topic=unfinished is not None,
                    ),
                    request_id=req_id,
                    conversation_id=identifier,
                    source_event_id=f"chat:{req_id}:conversation",
                    metadata={
                        "browser_invoked": browser_invoked,
                        "browser_succeeded": task_status
                        == TaskStatus.COMPLETED.value,
                    },
                )
            )
            if wrote_conversation:
                continuity_events_written += 1
            elif self.continuity_enabled:
                continuity_degraded = True
            if handoff and handoff_intent:
                handoff_summary = safe_bounded_summary(
                    "任务："
                    f"{safe_bounded_summary(str(handoff.metadata.get('current_task') or handoff.summary), limit=180)}；"
                    f"用户进度：{safe_bounded_summary(message, limit=180)}；"
                    f"夏夏结果：{safe_bounded_summary(generated.text, limit=250)}；"
                    + (
                        "已完成。"
                        if handoff_intent == "resolved"
                        else "下一步："
                        f"{safe_bounded_summary(str(handoff.metadata.get('next_step') or '继续当前任务'), limit=140)}。"
                    ),
                    limit=800,
                )
                wrote_handoff = await self._write_continuity_event(
                    self._base_event(
                        source=self.continuity_source,
                        event_type="topic_updated",
                        summary=handoff_summary,
                        importance=8,
                        status=handoff_intent,
                        request_id=req_id,
                        conversation_id=identifier,
                        source_event_id=f"chat:{req_id}:handoff",
                        metadata={
                            "topic_key": handoff.metadata["topic_key"],
                            "current_task": handoff.metadata.get("current_task")
                            or handoff.summary,
                            "next_step": (
                                handoff.metadata.get("next_step")
                                if handoff_intent == "active"
                                else None
                            ),
                        },
                    )
                )
                if wrote_handoff:
                    continuity_events_written += 1
                elif self.continuity_enabled:
                    continuity_degraded = True
            if unfinished and handoff_intent is None:
                wrote_unfinished = await self._write_continuity_event(
                    self._base_event(
                        source=self.continuity_source,
                        event_type="unfinished_topic",
                        summary=unfinished,
                        importance=8,
                        request_id=req_id,
                        conversation_id=identifier,
                        source_event_id=f"chat:{req_id}:unfinished",
                        metadata=(
                            {"category": unfinished_category}
                            if unfinished_category
                            else None
                        ),
                    )
                )
                if wrote_unfinished:
                    continuity_events_written += 1
                elif self.continuity_enabled:
                    continuity_degraded = True
            ledger_ok, finalized = await self._ledger_call(
                "finalize_request",
                self.execution_ledger.finalize_request,
                req_id,
                chat_status=ChatStatus.COMPLETED,
            )
            execution_ledger_degraded = execution_ledger_degraded or not ledger_ok
            request_finalized = ledger_ok
            if isinstance(finalized, TaskStatus):
                task_status = finalized.value
            status = "ok"
            return RuntimeResult(
                reply=generated.text,
                conversation_id=identifier,
                request_id=req_id,
                model=generated.model,
                retrieved_memory_count=retrieved_count,
                memory_degraded=degraded,
                browser_invoked=browser_invoked,
                browser_degraded=browser_degraded,
                browser_tool_calls=browser_tool_calls,
                browser_generation=browser_generation,
                browser_result_truncated=browser_result_truncated,
                browser_intent_decision=browser_intent_decision,
                destination_resolution_attempted=destination_resolution_attempted,
                browser_destination_url=browser_destination_url,
                browser_page_url=browser.page_url,
                browser_page_title=browser.page_title,
                browser_error_code=browser.error_code,
                continuity_degraded=continuity_degraded,
                continuity_capsule_event_count=continuity_capsule_event_count,
                continuity_capsule_char_count=continuity_capsule_char_count,
                continuity_events_written=continuity_events_written,
                task_status=task_status,
                chat_completed=True,
                execution_evidence_status=(
                    "unavailable" if execution_ledger_degraded else "available"
                ),
            )
        finally:
            final_chat_status = (
                ChatStatus.COMPLETED if status == "ok" else ChatStatus.FAILED
            )
            if not request_finalized:
                ledger_ok, finalized = await self._ledger_call(
                    "finalize_request",
                    self.execution_ledger.finalize_request,
                    req_id,
                    chat_status=final_chat_status,
                )
                execution_ledger_degraded = execution_ledger_degraded or not ledger_ok
                if isinstance(finalized, TaskStatus):
                    task_status = finalized.value
            logger.info(
                "chat_request_completed",
                extra={
                    "fields": {
                        "request_id": req_id,
                        "conversation_id": identifier,
                        "model_provider": self.provider.name,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                        "status": status,
                        "retrieved_memory_count": retrieved_count,
                        "memory_degraded": degraded,
                        "browser_invoked": browser_invoked,
                        "browser_degraded": browser_degraded,
                        "browser_tool_calls": browser_tool_calls,
                        "browser_generation": browser_generation,
                        "browser_result_truncated": browser_result_truncated,
                        "browser_intent_decision": browser_intent_decision,
                        "destination_resolution_attempted": (
                            destination_resolution_attempted
                        ),
                        "browser_destination_url": browser_destination_url,
                        "final_chat_completion": status == "ok",
                        "chat_status": final_chat_status.value,
                        "task_status": task_status,
                        "continuity_degraded": continuity_degraded,
                        "continuity_capsule_event_count": continuity_capsule_event_count,
                        "continuity_capsule_char_count": continuity_capsule_char_count,
                        "continuity_events_written": continuity_events_written,
                    }
                },
            )
