from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar
from urllib.parse import urlsplit, urlunsplit

from app.execution.ledger import ExecutionRequestSnapshot, VerificationState


@dataclass(frozen=True, slots=True)
class ProjectedExecutionClaim:
    action: str
    destinations: tuple[str, ...] = ()
    target: str | None = None
    count: int | None = None


@dataclass(frozen=True, slots=True)
class ExecutionClaimProjection:
    claims: tuple[ProjectedExecutionClaim, ...]
    all_completed: bool = False
    unknown_execution_success: bool = False

    @property
    def contains_execution_success(self) -> bool:
        return bool(self.claims) or self.all_completed or self.unknown_execution_success


@dataclass(frozen=True, slots=True)
class ClaimGateResult:
    text: str
    allowed: bool
    blocked_actions: tuple[str, ...] = ()
    projection: ExecutionClaimProjection | None = None


class ClaimGate:
    """Fail-closed authority for mapping final execution claims to ledger facts.

    Patterns only project possible claims. Authorization uses terminal,
    request-bound facts including action, URL/target, count, and whole-task status.
    """

    _ACTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "navigate",
            re.compile(
                r"(?:已(?:经)?|成功)?(?:打开|访问|导航到|进入|前往)|"
                r"\b(?:opened|visited|navigated\s+to|went\s+to|is\s+open)\b|"
                r"\bnavigate.{0,20}(?:success|complete|succeed|成功|完成)",
                re.IGNORECASE,
            ),
        ),
        (
            "read_page",
            re.compile(
                r"(?:已(?:经)?|成功)(?:读取|阅读|查看|检查)|页面已(?:经)?读取|"
                r"\b(?:read|inspected|examined)\s+(?:the\s+)?page\b",
                re.IGNORECASE,
            ),
        ),
        (
            "click",
            re.compile(
                r"(?:已(?:经)?|成功)(?:点击|点开)|"
                r"(?:点击|点开).{0,40}(?:已执行|已完成|成功)|\bclicked\b",
                re.IGNORECASE,
            ),
        ),
        (
            "type",
            re.compile(
                r"(?:已(?:经)?|成功)(?:输入|填入|写入)|"
                r"输入动作.{0,30}(?:已完成|成功)|"
                r"\b(?:typed|filled\s+in|entered)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "submit",
            re.compile(
                r"(?:已(?:经)?|成功)(?:提交|发送表单)|\bsubmitted\b",
                re.IGNORECASE,
            ),
        ),
        (
            "scroll",
            re.compile(
                r"(?:已(?:经)?滚动|滚动到(?:了)?|已到达页面(?:底部|顶部))|"
                r"\b(?:scrolled|reached\s+the\s+(?:bottom|top))\b",
                re.IGNORECASE,
            ),
        ),
        (
            "back",
            re.compile(
                r"(?:已(?:经)?后退|已返回上一页)|\bwent\s+back\b", re.IGNORECASE
            ),
        ),
        (
            "forward",
            re.compile(r"(?:已(?:经)?前进)|\bwent\s+forward\b", re.IGNORECASE),
        ),
        (
            "refresh",
            re.compile(r"(?:已(?:经)?刷新|已重新加载)|\brefreshed\b", re.IGNORECASE),
        ),
        (
            "close_tab",
            re.compile(
                r"(?:已(?:经)?关闭).{0,24}(?:标签页|页面|tab)|"
                r"\bclosed\s+(?:the\s+)?tabs?\b",
                re.IGNORECASE,
            ),
        ),
        (
            "switch_tab",
            re.compile(
                r"(?:已(?:经)?切换).{0,20}(?:标签页|tab)|\bswitched\s+tabs?\b",
                re.IGNORECASE,
            ),
        ),
        (
            "open_new_tab",
            re.compile(
                r"(?:已(?:经)?新建|已打开新).{0,12}标签页|"
                r"\bopened\s+a\s+new\s+tab\b",
                re.IGNORECASE,
            ),
        ),
        (
            "screenshot",
            re.compile(
                r"(?:已(?:经)?截图|已保存截图)|"
                r"\bcaptured\s+(?:a\s+)?screenshot\b",
                re.IGNORECASE,
            ),
        ),
        (
            "takeover_create",
            re.compile(
                r"(?:已(?:经)?(?:触发|创建|发起).{0,20}(?:接管|takeover)|"
                r"已(?:经)?接管|控制权已交给你|\btakeover\s+(?:started|created)\b|"
                r"\bcontrol\s+(?:was|has\s+been)\s+handed\s+(?:to|over))",
                re.IGNORECASE,
            ),
        ),
        (
            "page_state",
            re.compile(
                r"(?:当前|最终)(?:页面|网页)(?:是|为|位于)|"
                r"\b(?:current|final)\s+page\s+(?:is|at)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "takeover_revoke",
            re.compile(
                r"(?:已(?:经)?(?:释放|撤销|结束)接管)|"
                r"\btakeover\s+(?:released|revoked|ended)\b",
                re.IGNORECASE,
            ),
        ),
    )
    _ALL_COMPLETED = re.compile(
        r"(?:所有|全部|整个|本次).{0,20}(?:任务|操作|动作).{0,12}(?:已)?(?:完成|成功)|"
        r"\ball\s+(?:browser\s+)?(?:actions?|tasks?).{0,16}(?:completed|succeeded|successful)|"
        r"\beverything.{0,12}(?:completed|succeeded|done)\b",
        re.IGNORECASE,
    )
    _SUCCESS_CONTEXT = re.compile(
        r"(?:浏览器|browser|页面|page|标签页|tab|接管|takeover|控制权|"
        r"点击|输入|提交|滚动|导航|访问|打开|读取|刷新|后退|前进|执行|操作|动作|任务|"
        r"applied_verified)",
        re.IGNORECASE,
    )
    _SUCCESS_SIGNAL = re.compile(
        r"(?:已(?:经)?|成功|完成|达成|生效|交给|到达|呈现|就绪|加载|抵达|搞定|"
        r"completed|success(?:ful(?:ly)?)?|succeeded|done|executed|applied|"
        r"opened|visited|clicked|typed|submitted|scrolled|refreshed|handed|loaded)",
        re.IGNORECASE,
    )
    _NEGATIVE_ONLY = re.compile(
        r"(?:没有|并未|未曾|未能|不能|无法|不具备|不代表|"
        r"\bnot\b|\bdidn['’]?t\b|\bwasn['’]?t\b|\bcannot\b|\bcan['’]?t\b|"
        r"\bno\s+verified\b)",
        re.IGNORECASE,
    )
    _URL = re.compile(r"https?://[^\s<>()（）\[\]{}，。；、]+", re.IGNORECASE)
    _QUOTED_TARGET = re.compile(r"[“\"']([^”\"']{1,200})[”\"']")
    _NAVIGATE_TARGET = re.compile(
        r"(?:打开|访问|导航到|进入|前往)\s*(?:了|这个|该)?\s*([^，。；;！？!?\n]{1,100})|"
        r"\b(?:opened|visited|navigated\s+to|went\s+to)\s+([^,.;!?\n]{1,100})",
        re.IGNORECASE,
    )
    _GENERIC_NAVIGATE_TARGETS: ClassVar[set[str]] = {
        "",
        "它",
        "页面",
        "网页",
        "网站",
        "站点",
        "链接",
        "网址",
        "目标页面",
        "目标网页",
        "this page",
        "the page",
        "target page",
    }

    @staticmethod
    def _canonical_url(value: str) -> str | None:
        try:
            parsed = urlsplit(value.rstrip(".,;:!?，。；：！？"))
            if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
                return None
            port = parsed.port
        except ValueError:
            return None
        host = parsed.hostname.casefold()
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if port is not None and not (
            (parsed.scheme.casefold() == "http" and port == 80)
            or (parsed.scheme.casefold() == "https" and port == 443)
        ):
            host = f"{host}:{port}"
        return urlunsplit(
            (parsed.scheme.casefold(), host, parsed.path or "/", parsed.query, "")
        )

    @classmethod
    def project(cls, text: str) -> ExecutionClaimProjection:
        all_completed = bool(cls._ALL_COMPLETED.search(text))
        urls = tuple(
            value
            for match in cls._URL.findall(text)
            if (value := cls._canonical_url(match)) is not None
        )
        positioned_claims: list[tuple[int, ProjectedExecutionClaim]] = []
        projected_spans: list[tuple[int, int]] = []
        for action, pattern in cls._ACTION_PATTERNS:
            matches = tuple(pattern.finditer(text))
            if not matches:
                continue
            for match in matches:
                prefix = text[max(0, match.start() - 20) : match.start()]
                if cls._NEGATIVE_ONLY.search(prefix):
                    continue
                left = max(
                    text.rfind(mark, 0, match.start())
                    for mark in ("。", "；", ";", "\n")
                )
                right_candidates = [
                    position
                    for mark in ("。", "；", ";", "\n")
                    if (position := text.find(mark, match.end())) >= 0
                ]
                right = min(right_candidates) if right_candidates else len(text)
                clause = text[left + 1 : right]
                if action == "navigate" and re.search(
                    r"新.{0,8}标签页|new\s+tab", clause, re.IGNORECASE
                ):
                    continue
                count: int | None = None
                if re.search(
                    r"(?:两次|2\s*次|twice|two\s+times)", clause, re.IGNORECASE
                ):
                    count = 2
                else:
                    numeric = re.search(
                        r"\b(\d{1,3})\s+times\b|(\d{1,3})\s*次",
                        clause,
                        re.IGNORECASE,
                    )
                    if numeric:
                        count = int(numeric.group(1) or numeric.group(2))
                local_urls: set[str] = set()
                for raw_url in cls._URL.findall(text[left + 1 : right]):
                    if (value := cls._canonical_url(raw_url)) is not None:
                        local_urls.add(value)
                if not local_urls and len(urls) == 1:
                    local_urls.add(urls[0])
                quoted = cls._QUOTED_TARGET.search(clause)
                navigate_target: str | None = None
                if action == "navigate" and not local_urls:
                    target_match = cls._NAVIGATE_TARGET.search(clause)
                    if target_match:
                        navigate_target = (
                            target_match.group(1) or target_match.group(2)
                        ).strip()
                        navigate_target = re.sub(
                            r"^(?:并|and\s+)?(?:已)?(?:读取|阅读|查看|检查|read|inspect)\s*",
                            "",
                            navigate_target,
                            flags=re.IGNORECASE,
                        ).strip()
                        if navigate_target.casefold() in cls._GENERIC_NAVIGATE_TARGETS:
                            navigate_target = None
                positioned_claims.append(
                    (
                        match.start(),
                        ProjectedExecutionClaim(
                            action=action,
                            destinations=(
                                tuple(sorted(local_urls))
                                if action in {"navigate", "read_page", "page_state"}
                                else ()
                            ),
                            target=(
                                quoted.group(1).strip()
                                if quoted and action == "click"
                                else navigate_target
                            ),
                            count=count,
                        ),
                    )
                )
                projected_spans.append(match.span())
        claims = tuple(
            item for _, item in sorted(positioned_claims, key=lambda item: item[0])
        )
        residual = list(text)
        for start, end in projected_spans:
            residual[start:end] = " " * (end - start)
        for match in cls._ALL_COMPLETED.finditer(text):
            residual[match.start() : match.end()] = " " * (match.end() - match.start())
        residual_text = "".join(residual)
        positive_context = bool(
            cls._SUCCESS_CONTEXT.search(residual_text)
            and cls._SUCCESS_SIGNAL.search(residual_text)
        )
        negative_only = bool(cls._NEGATIVE_ONLY.search(text)) and not bool(claims)
        return ExecutionClaimProjection(
            claims=claims,
            all_completed=all_completed,
            unknown_execution_success=(
                positive_context
                and not claims
                and not all_completed
                and not negative_only
            ),
        )

    @staticmethod
    def _verified_facts(
        snapshot: ExecutionRequestSnapshot,
    ) -> tuple[dict[str, object], ...]:
        sources = {
            str(item.get("action_id")): str(item.get("source"))
            for item in snapshot.subgoals
        }
        return tuple(
            {**item, "subgoal_source": sources.get(str(item.get("action_id")))}
            for item in snapshot.final_attempts
            if item.get("normalized_action") != "status"
            and item.get("verification_state")
            in {
                VerificationState.APPLIED_VERIFIED.value,
                VerificationState.CAPTURED_NOT_INTERPRETED.value,
            }
        )

    @classmethod
    def _matching_facts(
        cls,
        claim: ProjectedExecutionClaim,
        facts: tuple[dict[str, object], ...],
    ) -> tuple[dict[str, object], ...]:
        aliases = {
            "open_new_tab": {"new_tab"},
            "takeover_create": {"takeover_create"},
            "takeover_revoke": {"takeover_revoke"},
        }
        if claim.action == "page_state":
            return tuple(
                item
                for item in facts
                if item.get("actual_final_url") or item.get("actual_title")
            )
        actions = aliases.get(claim.action, {claim.action})
        return tuple(
            item for item in facts if str(item.get("normalized_action")) in actions
        )

    @classmethod
    def _claim_matches(
        cls,
        claim: ProjectedExecutionClaim,
        facts: tuple[dict[str, object], ...],
    ) -> bool:
        matches = cls._matching_facts(claim, facts)
        if not matches:
            return False
        if claim.count is not None:
            required_matches = tuple(
                item
                for item in matches
                if item.get("subgoal_source") == "required_task"
            )
            countable = required_matches or matches
            if len(countable) != claim.count:
                return False
        if claim.destinations:
            actual_urls = {
                value
                for item in matches
                for raw in (item.get("actual_final_url"), item.get("destination"))
                if isinstance(raw, str)
                and (value := cls._canonical_url(raw)) is not None
            }
            if not set(claim.destinations).issubset(actual_urls):
                return False
        if claim.target:
            expected = claim.target.casefold().strip()
            if claim.action == "navigate":
                normalized_expected = re.sub(
                    r"(?:登录|登陆|主页|首页|页面|网页|网站|站点|链接|网址|"
                    r"login|sign.?in|home|page|website|site|link)",
                    "",
                    expected,
                    flags=re.IGNORECASE,
                ).strip()
                actual_values = {
                    str(value).casefold().strip()
                    for item in matches
                    for value in (
                        item.get("actual_final_url"),
                        item.get("actual_title"),
                        item.get("destination"),
                    )
                    if isinstance(value, str) and value.strip()
                }
                return any(
                    expected in value
                    or (
                        normalized_expected
                        and normalized_expected
                        in re.sub(
                            r"(?:登录|登陆|主页|首页|页面|网页|网站|站点|链接|网址|"
                            r"login|sign.?in|home|page|website|site|link)",
                            "",
                            value,
                            flags=re.IGNORECASE,
                        )
                    )
                    for value in actual_values
                )
            actual_targets = {
                str(value).casefold().strip()
                for item in matches
                for target in (item.get("target"),)
                if isinstance(target, dict)
                for value in target.values()
                if isinstance(value, str) and value.strip()
            }
            if expected not in actual_targets:
                return False
        return True

    @classmethod
    def _claims_follow_execution_order(
        cls,
        claims: tuple[ProjectedExecutionClaim, ...],
        facts: tuple[dict[str, object], ...],
    ) -> bool:
        ordered_facts = tuple(
            sorted(facts, key=lambda item: int(item.get("call_sequence") or 0))
        )
        cursor = -1
        for claim in claims:
            repetitions = claim.count or 1
            single = ProjectedExecutionClaim(
                action=claim.action,
                destinations=claim.destinations,
                target=claim.target,
            )
            for _ in range(repetitions):
                search_start = cursor if claim.action == "page_state" else cursor + 1
                matched_index = next(
                    (
                        index
                        for index in range(max(0, search_start), len(ordered_facts))
                        if cls._claim_matches(single, (ordered_facts[index],))
                    ),
                    None,
                )
                if matched_index is None:
                    return False
                cursor = matched_index
        return True

    @staticmethod
    def unavailable_summary(browser_error_code: str | None = None) -> str:
        if browser_error_code:
            return (
                f"Browser 返回错误 {browser_error_code}；但本次执行台账不可用，"
                "因此无法验证任何成功操作。"
            )
        return "本次执行台账不可用，因此无法验证或声称任何 Browser 操作成功。"

    @classmethod
    def gate(
        cls,
        text: str,
        snapshot: ExecutionRequestSnapshot | None,
        *,
        evidence_available: bool = True,
        execution_context: bool = False,
        historical: bool = False,
        browser_error_code: str | None = None,
    ) -> ClaimGateResult:
        projection = cls.project(text)
        if (
            execution_context
            and not projection.contains_execution_success
            and cls._SUCCESS_SIGNAL.search(text)
            and not cls._NEGATIVE_ONLY.search(text)
        ):
            projection = ExecutionClaimProjection(
                claims=projection.claims,
                all_completed=projection.all_completed,
                unknown_execution_success=True,
            )
        if historical:
            if not evidence_available or snapshot is None:
                return ClaimGateResult(
                    text="没有紧邻上一请求的精确执行台账；本回答没有新调用 Browser。",
                    allowed=False,
                    blocked_actions=("__historical_evidence_unavailable__",),
                    projection=projection,
                )
            return ClaimGateResult(text=text, allowed=True, projection=projection)
        if not evidence_available:
            if execution_context or projection.contains_execution_success:
                return ClaimGateResult(
                    text=cls.unavailable_summary(browser_error_code),
                    allowed=False,
                    blocked_actions=("__execution_evidence_unavailable__",),
                    projection=projection,
                )
            return ClaimGateResult(text=text, allowed=True, projection=projection)
        if not projection.contains_execution_success:
            return ClaimGateResult(text=text, allowed=True, projection=projection)
        if snapshot is None:
            return ClaimGateResult(
                text=cls.truthful_summary(None),
                allowed=False,
                blocked_actions=("__missing_current_request_evidence__",),
                projection=projection,
            )
        facts = cls._verified_facts(snapshot)
        blocked = [
            claim.action
            for claim in projection.claims
            if not cls._claim_matches(claim, facts)
        ]
        if projection.claims and not cls._claims_follow_execution_order(
            projection.claims, facts
        ):
            blocked.append("__claim_order__")
        if projection.all_completed and snapshot.task_status != "completed":
            blocked.append("__all_completed__")
        if projection.unknown_execution_success:
            blocked.append("__unprojectable_execution_success__")
        if not blocked:
            return ClaimGateResult(text=text, allowed=True, projection=projection)
        return ClaimGateResult(
            text=cls.truthful_summary(snapshot),
            allowed=False,
            blocked_actions=tuple(sorted(set(blocked))),
            projection=projection,
        )

    @classmethod
    def truthful_summary(cls, snapshot: ExecutionRequestSnapshot | None) -> str:
        if snapshot is None:
            return "本次没有可验证的 Browser 执行记录，因此不能声称任何浏览器操作已经发生。"
        required_ids = {
            str(item.get("action_id"))
            for item in snapshot.subgoals
            if item.get("source") == "required_task"
        }
        final_attempts = snapshot.final_attempts
        verified = [
            item
            for item in final_attempts
            if item.get("action_id") in required_ids
            and item.get("verification_state")
            in {
                VerificationState.APPLIED_VERIFIED.value,
                VerificationState.CAPTURED_NOT_INTERPRETED.value,
            }
        ]
        attempted_ids = {str(item.get("action_id")) for item in final_attempts}
        missing = [
            str(item.get("normalized_action"))
            for item in snapshot.subgoals
            if item.get("required") is True
            and str(item.get("action_id")) not in attempted_ids
        ]
        failures = [
            item
            for item in final_attempts
            if item.get("action_id") in required_ids
            and item.get("verification_state")
            not in {
                VerificationState.APPLIED_VERIFIED.value,
                VerificationState.CAPTURED_NOT_INTERPRETED.value,
            }
        ]
        parts = [
            f"request_id={snapshot.request_id} 的任务状态为 {snapshot.task_status}。"
        ]
        if verified:
            descriptions = []
            for item in verified:
                description = str(item.get("normalized_action"))
                if item.get("actual_final_url"):
                    description += f"({item['actual_final_url']})"
                descriptions.append(description)
            parts.append("已验证执行：" + "、".join(descriptions) + "。")
        if failures:
            parts.append(
                "未验证或失败："
                + "、".join(
                    f"{item.get('normalized_action')}({item.get('error_code') or item.get('verification_state')})"
                    for item in failures
                )
                + "。"
            )
        if missing:
            parts.append("未执行：" + "、".join(missing) + "。")
        if not verified and not failures and not missing:
            parts.append("没有当前请求的已验证 Browser 动作。")
        return "".join(parts)
