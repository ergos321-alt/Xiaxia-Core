from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Literal

BrowserActionKind = Literal[
    "navigate",
    "new_tab",
    "read_page",
    "tabs",
    "click",
    "type",
    "submit",
    "scroll",
    "seek",
    "back",
    "forward",
    "refresh",
    "switch_tab",
    "close_tab",
    "screenshot",
    # Phase 2B keeps future authenticated intents representable without
    # claiming that the current Browser runtime can execute them yet.
    "like",
    "favorite",
    "follow",
    "comment",
    "share",
    "private_message",
]

IntentSource = Literal["user", "self", "runtime", "continuation"]
BrowserTaskKind = Literal[
    "browser_task",
    "open_ended_browse",
    "historical_evidence_query",
    "execution_status_query",
    "non_browser_response",
    "none",
]


@dataclass(frozen=True, slots=True)
class BrowserDestination:
    url: str | None
    site_hint: str | None
    source: Literal["explicit_url", "explicit_domain", "named_site", "model"]
    position: int


@dataclass(frozen=True, slots=True)
class BrowserAction:
    kind: BrowserActionKind
    position: int
    destination_index: int | None = None
    semantic_target: str | None = None
    value: str | None = None
    ordinal: int = 0
    role: str | None = None
    success_condition: str = "tool_result_normalized"
    failure_code: str = "browser_action_failed"
    required: bool = True
    depends_on_previous: bool = True
    subgoal_id: str | None = None
    source_span: tuple[int, int] | None = None
    dependency_ids: tuple[str, ...] = ()
    condition: str | None = None


@dataclass(frozen=True, slots=True)
class BrowserConstraint:
    """A source-bound rule that may restrict planning or execution."""

    constraint_id: str
    kind: Literal["prohibit_action", "prohibit_browser", "safety"]
    polarity: Literal["prohibit", "require"]
    action: str | None = None
    target: str | None = None
    source_span: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class BrowserIntent:
    """Versioned semantic task representation, independent of intent origin."""

    version: str
    source: IntentSource
    task_kind: BrowserTaskKind
    destinations: tuple[BrowserDestination, ...] = ()
    page_scope: Literal[
        "current", "contextual", "destination", "multiple", "unspecified"
    ] = "unspecified"
    historical_reference: Literal["previous_request", "current_request"] | None = None
    constraints: tuple[BrowserConstraint, ...] = ()
    required_subgoals: tuple[BrowserAction, ...] = ()
    ambiguity: str | None = None
    requires_destination_resolution: bool = False

    @property
    def browser_prohibited(self) -> bool:
        return any(item.kind == "prohibit_browser" for item in self.constraints)

    def to_action_plan(self) -> BrowserActionPlan:
        compatibility_constraints = (
            "public_http_only",
            "bounded_execution",
            "no_transaction_without_confirmation",
            *(f"prohibit:{item.action or 'browser'}" for item in self.constraints),
        )
        return BrowserActionPlan(
            destinations=self.destinations,
            active_page_scope=(
                self.page_scope if self.page_scope != "unspecified" else "current"
            ),
            actions=self.required_subgoals,
            tab_policy=(
                "mixed"
                if any(item.kind == "new_tab" for item in self.required_subgoals)
                else "current"
            ),
            constraints=tuple(compatibility_constraints),
            semantic_constraints=self.constraints,
            version=self.version,
            source=self.source,
            task_kind=self.task_kind,
            historical_reference=self.historical_reference,
            ambiguity=self.ambiguity,
            requires_destination_resolution=self.requires_destination_resolution,
        )


@dataclass(frozen=True, slots=True)
class BrowserActionPlan:
    destinations: tuple[BrowserDestination, ...] = ()
    active_page_scope: Literal["current", "contextual", "destination", "multiple"] = (
        "current"
    )
    actions: tuple[BrowserAction, ...] = ()
    tab_policy: Literal["current", "new_tab", "mixed"] = "current"
    constraints: tuple[str, ...] = (
        "public_http_only",
        "bounded_execution",
        "no_transaction_without_confirmation",
    )
    evidence_requirements: tuple[str, ...] = (
        "normalized_tool_result",
        "final_url_for_navigation",
        "post_action_verification",
    )
    ambiguity: str | None = None
    requires_destination_resolution: bool = False
    semantic_constraints: tuple[BrowserConstraint, ...] = ()
    version: str = "browser-intent/1"
    source: IntentSource = "user"
    task_kind: BrowserTaskKind = "browser_task"
    historical_reference: Literal["previous_request", "current_request"] | None = None

    @property
    def actionable(self) -> bool:
        return (
            bool(self.actions)
            and self.ambiguity is None
            and all(item.url is not None for item in self.destinations)
        )


class BrowserPlanningBoundary:
    """Build one ordered, inspectable plan before any Browser action runs."""

    _URL = re.compile(r"https?://[^\s<>\"'，。！？；、（）【】《》]+", re.IGNORECASE)
    _DOMAIN = re.compile(
        r"(?<![@A-Za-z0-9_])(?:www\.)?(?:[a-z0-9-]+\.)+(?:com|org|net|cn|io|ai|dev|co)"
        r"(?:/[^\s<>\"'，。！？；、（）【】《》]*)?",
        re.IGNORECASE,
    )
    _OPEN = re.compile(
        r"打开|访问|浏览|进入|前往|去|上|"
        r"\b(?:open|visit|browse|go\s+to)\b",
        re.IGNORECASE,
    )
    _READ = re.compile(
        r"看看|看一下|读取|读一下|检查|内容|写了什么|\b(?:read|inspect|check)\b",
        re.IGNORECASE,
    )
    _READ_ACTION = re.compile(
        r"看看|看一下|读取|阅读|读一下|检查|"
        r"\b(?:read|inspect|check)(?:\s+(?:the|this|current))?\b",
        re.IGNORECASE,
    )
    _CURRENT = re.compile(
        r"当前(?:页面|网页|标签页)?|这个(?:页面|网页)|这个网页|这里|"
        r"刚才那个页面|刚才的页面|\b(?:current|this)\s+(?:page|tab)\b|\bhere\b",
        re.IGNORECASE,
    )
    _CONFLICT = re.compile(
        r"(?:^|\s)(?:or|either)(?:\s|$)|或(?:者)?|还是|任选", re.IGNORECASE
    )
    _NEW_TAB = re.compile(r"新建|新开|新标签|新 tab|new\s+tab", re.IGNORECASE)
    _BACK = re.compile(
        r"后退|返回上一页|\bbrowser\s+back\b|\bgo\s+back\b", re.IGNORECASE
    )
    _NEGATED_REFERENCE = re.compile(
        r"(?:不要|别|避免|排除|禁止|不再|不要再)(?:\s|去|访问|打开|浏览)*$|"
        r"\b(?:do\s+not|don't|never|avoid|exclude|except|not)\b[^，。！？；,!?;]{0,24}$",
        re.IGNORECASE,
    )
    _CLICK_ACTION = re.compile(
        r"(?:点击(?!后(?:的|页面|结果))|点开|按下|\bclick(?:\s+on)?\b)\s*"
        r"(?P<direct>[^，。！？；、;\n]+)|"
        r"(?:找到|查找|寻找|定位|find|locate)\s*"
        r"(?:并(?:真正)?|然后|and\s+then|and)?\s*"
        r"(?:进入|打开|点开|点击|选择|跟随|enter|open|click|select|follow)\s*"
        r"(?P<found>[^，。！？；、;\n]+)",
        re.IGNORECASE,
    )
    _FOLLOWUP_ENTRY_ACTION = re.compile(
        r"(?:^|[，,；;。.!！?？\n]|然后|接着|随后|再)\s*"
        r"(?:进入|选择|打开|open|enter|select)\s*"
        r"(?P<target>[^，。！？；、;\n]+)",
        re.IGNORECASE,
    )
    _LOCATE_ACTION = re.compile(
        r"(?:找到|查找|寻找|定位|find|locate)\s*"
        r"(?P<target>[^，。！？；、;\n]+?)\s*"
        r"(?=并(?:读取|阅读|看看|检查)|然后|接着|随后|[，。！？；、;\n]|$)",
        re.IGNORECASE,
    )
    _PROHIBITION = re.compile(
        r"(?:不要|别|禁止|避免|不许|无需|不得)\s*(?:再|重新)?\s*"
        r"(?P<zh_action>调用浏览器|使用浏览器|点击|点开|查找|寻找|搜索|检索|"
        r"打开|访问|浏览|输入|填写|提交|滚动|后退|前进|刷新)|"
        r"\b(?:do\s+not|don't|never|avoid|must\s+not)\s+(?:re-?)?"
        r"(?P<en_action>use\s+(?:the\s+)?browser|click|find|search|open|visit|browse|"
        r"type|input|submit|scroll|go\s+back|go\s+forward|refresh)\b",
        re.IGNORECASE,
    )
    _STATUS_QUESTION = re.compile(
        r"(?:刚才|刚刚|上一(?:个)?(?:请求|任务)|之前).{0,80}"
        r"(?:所有操作|做到哪|哪一步|失败|成功|完成|有没有|是否|实际执行)|"
        r"(?:哪一步失败|刚才做到哪|刚才有没有点击)|"
        r"(?:之前|刚才|刚刚|以前|历史).{0,48}(?:浏览|打开|访问|点击|浏览器)"
        r".{0,32}(?:模拟|真实|实际|吗|么|是否|是不是)|"
        r"\b(?:previous|last)\s+(?:request|task).{0,80}"
        r"(?:complete|completed|succeed|failed|execute|actually|which\s+step|what\s+did)\b",
        re.IGNORECASE,
    )
    _CURRENT_REQUEST_STATUS = re.compile(
        r"(?:这一条|本条|当前)(?:请求)?|\b(?:this|current)\s+request\b",
        re.IGNORECASE,
    )
    _NO_BROWSER = re.compile(
        r"(?:不要|不许|无需|不得).{0,16}(?:调用|使用).{0,8}浏览器|"
        r"\b(?:do\s+not|don't|without)\s+(?:call|use|invoking).{0,12}browser\b",
        re.IGNORECASE,
    )
    _OPEN_ENDED = re.compile(
        r"(?:自己|随便|自由|自主).{0,18}(?:上网|浏览|逛|看看)|"
        r"(?:想看什么就看什么|上网玩一会儿)|"
        r"(?:去|开始|继续|所以去|再出去|再)?\s*逛(?:一逛|逛|会儿|一下|一次)?|"
        r"(?:换|另选|重新选|挑)(?:一个|个)?(?:别的|不同的|其他)?(?:网站|网页|地方)|"
        r"\b(?:browse\s+around|explore\s+the\s+web|browse\s+whatever|pick\s+(?:a\s+)?site)\b",
        re.IGNORECASE,
    )
    _CONTINUATION_SCROLL = re.compile(
        r"(?:继续|接着)?\s*(?:往下|向下)(?:看|浏览)|"
        r"\b(?:continue|keep)\s+(?:down|scrolling)\b",
        re.IGNORECASE,
    )
    _DESTINATION_REQUEST = re.compile(
        r"(?:^|[。！？；!?;\n])[^，。！？；,!?;\n]{0,32}?"
        r"(?:打开|访问|前往|去|上(?!网)|浏览(?!器)|\bopen\b|\bvisit\b|\bgo\s+to\b|\bbrowse\b)\s*"
        r"(?P<target>[^，。！？；,!?;\n]{1,120})",
        re.IGNORECASE,
    )
    _NAMED_DESTINATION_REQUEST = re.compile(
        r"(?:打开|访问|前往|去|上(?!网)|浏览(?!器)|\bopen\b|\bvisit\b|"
        r"\bgo\s+to\b|\bbrowse\b)\s*"
        r"(?P<target>.+?)"
        r"(?=(?:看看|看一下|逛逛|逛一逛|检查|读取|阅读)|"
        r"(?:\s+(?:新建\s*tab|新开标签页|然后|再|接着|随后)?\s*"
        r"(?:打开|访问|前往|去|浏览|\bopen\b|\bvisit\b|\bgo\s+to\b|\bbrowse\b))|"
        r"[，。！？；,!?;\n]|$)",
        re.IGNORECASE,
    )
    _AUTH_TASK = re.compile(
        r"登录|登陆|验证码|密码|账号|human\s+takeover|login|log\s*in|sign\s*in",
        re.IGNORECASE,
    )
    _TAKEOVER_TASK = re.compile(
        r"human\s+takeover|发起.{0,12}接管|让我接管|交给我接管|"
        r"\bhand\s+(?:browser\s+)?control\s+to\s+me\b",
        re.IGNORECASE,
    )
    _CAPABILITY_QUERY = re.compile(
        r"(?:你|xiaxia).{0,16}(?:有|能|支持).{0,16}(?:浏览器|上网|浏览网页)|"
        r"\bcan\s+(?:you|xiaxia)\s+(?:browse|use\s+a\s+browser)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _clean_destination_url(value: str) -> str:
        cleaned = value.rstrip("。！？；，,.!?;)]}")
        return "".join(
            character
            for character in cleaned
            if unicodedata.category(character) != "Cf" and character != "\ufffd"
        )

    @staticmethod
    def _clean_target(value: str) -> str:
        cleaned = re.split(
            r"(?:然后|接着|随后|并且|停止|再(?:读取|看看|检查)|\b(?:then|stop)\b)",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        return cleaned.strip(" ：:，,。.!！？?\"'‘’“”《》「」")[:500]

    @staticmethod
    def _explicit_target_role(value: str) -> str | None:
        if re.search(r"链接|\blinks?\b", value, re.IGNORECASE):
            return "link"
        return None

    @classmethod
    def _is_result_report_destination(
        cls, message: str, match: re.Match[str]
    ) -> bool:
        """Reject result-report wording such as “告诉我打开了什么”."""

        target = cls._clean_target(match.group("target"))
        clause = cls._clause_before(message, match.start())
        return bool(
            re.search(
                r"告诉我|告知我|说说|说明|报告|\btell\s+me\b",
                clause,
                re.IGNORECASE,
            )
            and re.fullmatch(
                r"(?:了|的是|了的)?(?:什么|哪里|哪(?:个|一)?(?:页|页面|网站)?)",
                target,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _ordinal(value: str | None) -> int:
        if not value:
            return 0
        mapping = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
        number = mapping.get(value, int(value) if value.isdigit() else 1)
        return max(0, number - 1)

    @classmethod
    def _is_negated_reference(cls, message: str, position: int) -> bool:
        return bool(
            cls._NEGATED_REFERENCE.search(message[max(0, position - 48) : position])
        )

    @staticmethod
    def _clause_before(message: str, position: int, lower_bound: int = -1) -> str:
        start = max(
            lower_bound,
            max(
                message.rfind(separator, 0, position)
                for separator in (
                    "，",
                    "。",
                    "！",
                    "？",
                    "；",
                    ",",
                    ".",
                    "!",
                    "?",
                    ";",
                    "\n",
                )
            ),
        )
        return message[start + 1 : position]

    @staticmethod
    def _prohibited_action_name(value: str) -> str:
        normalized = re.sub(r"\s+", " ", value.casefold()).strip()
        mapping = {
            "调用浏览器": "browser",
            "使用浏览器": "browser",
            "use browser": "browser",
            "use the browser": "browser",
            "点击": "click",
            "点开": "click",
            "click": "click",
            "查找": "search",
            "寻找": "search",
            "搜索": "search",
            "检索": "search",
            "find": "search",
            "search": "search",
            "打开": "navigate",
            "访问": "navigate",
            "浏览": "navigate",
            "open": "navigate",
            "visit": "navigate",
            "browse": "navigate",
            "输入": "type",
            "填写": "type",
            "type": "type",
            "input": "type",
            "提交": "submit",
            "submit": "submit",
            "滚动": "scroll",
            "scroll": "scroll",
            "后退": "back",
            "go back": "back",
            "前进": "forward",
            "go forward": "forward",
            "刷新": "refresh",
            "refresh": "refresh",
        }
        return mapping.get(normalized, normalized.replace(" ", "_"))

    @classmethod
    def _semantic_constraints(cls, message: str) -> tuple[BrowserConstraint, ...]:
        constraints: list[BrowserConstraint] = []
        for match in cls._PROHIBITION.finditer(message):
            raw_action = match.group("zh_action") or match.group("en_action") or ""
            action = cls._prohibited_action_name(raw_action)
            remainder = message[match.end() :]
            target_match = re.match(
                r"\s*(?P<target>[^，。！？；,!?;\n]{0,120})", remainder
            )
            target = (
                cls._clean_target(target_match.group("target")) if target_match else ""
            )
            constraints.append(
                BrowserConstraint(
                    constraint_id=f"constraint-{len(constraints) + 1:03d}",
                    kind="prohibit_browser"
                    if action == "browser"
                    else "prohibit_action",
                    polarity="prohibit",
                    action=None if action == "browser" else action,
                    target=target or None,
                    source_span=match.span(),
                )
            )
        return tuple(constraints)

    @staticmethod
    def _position_is_prohibited(
        position: int, action: str, constraints: tuple[BrowserConstraint, ...]
    ) -> bool:
        for constraint in constraints:
            if constraint.kind == "prohibit_browser":
                return True
            if constraint.action != action or constraint.source_span is None:
                continue
            start, end = constraint.source_span
            if start <= position <= end:
                return True
            # The action token is inside the prohibition match; tolerate a short
            # modifier such as “重新” between the two.
            if start <= position <= end + 8:
                return True
        return False

    @staticmethod
    def _with_stable_subgoal_ids(
        actions: tuple[BrowserAction, ...] | list[BrowserAction],
    ) -> tuple[BrowserAction, ...]:
        ordered = sorted(actions, key=lambda item: item.position)
        result: list[BrowserAction] = []
        previous_id: str | None = None
        for index, action in enumerate(ordered, start=1):
            subgoal_id = action.subgoal_id or f"subgoal-{index:03d}"
            dependency_ids = (
                (previous_id,)
                if action.depends_on_previous and previous_id is not None
                else action.dependency_ids
            )
            result.append(
                replace(
                    action,
                    subgoal_id=subgoal_id,
                    source_span=action.source_span
                    or (action.position, action.position + 1),
                    dependency_ids=dependency_ids,
                )
            )
            previous_id = subgoal_id
        return tuple(result)

    @classmethod
    def destinations(cls, message: str) -> tuple[BrowserDestination, ...]:
        found: list[BrowserDestination] = []
        occupied: list[tuple[int, int]] = []
        for match in cls._URL.finditer(message):
            if cls._is_negated_reference(message, match.start()):
                occupied.append(match.span())
                continue
            url = cls._clean_destination_url(match.group(0))
            found.append(BrowserDestination(url, None, "explicit_url", match.start()))
            occupied.append(match.span())
        for match in cls._DOMAIN.finditer(message):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            if cls._is_negated_reference(message, match.start()):
                occupied.append(match.span())
                continue
            value = cls._clean_destination_url(match.group(0))
            found.append(
                BrowserDestination(
                    f"https://{value}", None, "explicit_domain", match.start()
                )
            )
            occupied.append(match.span())

        for match in cls._NAMED_DESTINATION_REQUEST.finditer(message):
            target_span = match.span("target")
            if cls._is_result_report_destination(message, match):
                continue
            # "find and open X" describes an in-page target, not a new site.
            # Keep that semantic action in the ordered subgoal parser instead of
            # manufacturing a second destination from the verb "open".
            if re.search(
                r"(?:\bfind\s+and|\blocate\s+and)\s*$",
                message[max(0, match.start() - 32) : match.start()],
                re.IGNORECASE,
            ):
                continue
            if any(
                start < target_span[1] and target_span[0] < end
                for start, end in occupied
            ):
                continue
            if cls._is_negated_reference(message, match.start()):
                continue
            hint = cls._clean_target(match.group("target"))
            if cls._is_adjectival_open_target(hint):
                # In phrases such as "打开的 tab" and "打开的页面", 打开 is
                # describing existing browser state rather than requesting a
                # destination. Non-navigation actions must not be sent through
                # destination resolution merely because this adjective occurs.
                continue
            hint = re.sub(
                r"^(?:新建\s*tab|新开标签页|然后|再|接着|随后)\s*",
                "",
                hint,
                flags=re.IGNORECASE,
            ).strip()
            if hint.casefold().startswith(("上网", "网上")):
                # “去上网逛逛” selects the open-ended browsing mode; “上网” is
                # not itself a named website that needs destination resolution.
                continue
            if (
                cls._OPEN_ENDED.fullmatch(hint)
                or re.fullmatch(
                    r"(?:哪(?:里|儿)?|什么(?:网站|网页|地方)?|where|whatever)",
                    hint,
                    re.IGNORECASE,
                )
                or hint.startswith("、")
            ):
                # Open-ended choice words and punctuation-led action lists are
                # not named destinations. The upper model chooses the site in
                # the open-ended browsing path.
                continue
            if not hint or cls._URL.fullmatch(hint) or cls._DOMAIN.fullmatch(hint):
                continue
            found.append(BrowserDestination(None, hint, "model", target_span[0]))

        found.sort(key=lambda item: item.position)
        deduplicated: list[BrowserDestination] = []
        for item in found:
            if not any(
                existing.url == item.url and existing.position == item.position
                for existing in deduplicated
            ):
                deduplicated.append(item)
        return tuple(deduplicated)

    @staticmethod
    def _is_adjectival_open_target(value: str) -> bool:
        return bool(
            re.match(
                r"^的\s*(?:页面|网页|标签页|tabs?\b)",
                value.strip(),
                re.IGNORECASE,
            )
        )

    @classmethod
    def _unresolved_destination_request(cls, message: str) -> re.Match[str] | None:
        return next(
            (
                match
                for match in cls._DESTINATION_REQUEST.finditer(message)
                if not cls._is_adjectival_open_target(match.group("target"))
                and not cls._is_result_report_destination(message, match)
            ),
            None,
        )

    @classmethod
    def _build_candidate_plan(cls, message: str) -> BrowserActionPlan:
        destinations = cls.destinations(message)
        actions: list[BrowserAction] = []

        if any(
            item.site_hint and cls._CONFLICT.search(item.site_hint)
            for item in destinations
        ):
            return BrowserActionPlan(
                destinations=destinations,
                active_page_scope="multiple",
                ambiguity="conflicting_destinations",
            )

        if len(destinations) > 1:
            for left, right in pairwise(destinations):
                if cls._CONFLICT.search(message[left.position : right.position]):
                    return BrowserActionPlan(
                        destinations=destinations,
                        active_page_scope="multiple",
                        ambiguity="conflicting_destinations",
                    )

        for index, destination in enumerate(destinations):
            clause = cls._clause_before(
                message,
                destination.position,
                destinations[index - 1].position if index else -1,
            )
            if cls._BACK.search(clause) or not cls._OPEN.search(clause):
                continue
            kind: BrowserActionKind = (
                "new_tab" if cls._NEW_TAB.search(clause) else "navigate"
            )
            actions.append(
                BrowserAction(
                    kind,
                    destination.position,
                    destination_index=index,
                    success_condition="final_url_observed",
                    failure_code="navigation_failed",
                )
            )

        navigated = {
            item.destination_index
            for item in actions
            if item.kind in {"navigate", "new_tab"}
        }
        if (
            len(destinations) > 1
            and len(navigated) == 1
            and all(item.source == "named_site" for item in destinations)
        ):
            # A second named entity without its own navigation verb is a
            # resource qualifier (for example, an organization on a code host),
            # not a second destination.
            return BrowserActionPlan(requires_destination_resolution=True)

        for click in cls._CLICK_ACTION.finditer(message):
            raw_target = click.group("direct") or click.group("found") or ""
            target = cls._clean_target(raw_target)
            ordinal = 0
            actions.append(
                BrowserAction(
                    "click",
                    click.start(),
                    semantic_target=target,
                    ordinal=ordinal,
                    role=cls._explicit_target_role(raw_target),
                    success_condition="physical_click_and_target_identity",
                    failure_code="element_not_interactable",
                )
            )

        for entry in cls._FOLLOWUP_ENTRY_ACTION.finditer(message):
            target = cls._clean_target(entry.group("target") or "")
            if (
                not target
                or re.match(
                    r"(?:新|当前|这个|该)?(?:页面|网页|标签页)(?:后|里|中|上|$)",
                    target,
                    re.IGNORECASE,
                )
                or any(
                    item.kind == "click" and abs(item.position - entry.start()) < 8
                    for item in actions
                )
                or any(
                    entry.start() <= destination.position < entry.end()
                    for destination in destinations
                )
            ):
                continue
            actions.append(
                BrowserAction(
                    "click",
                    entry.start(),
                    semantic_target=target,
                    success_condition="physical_click_and_target_identity",
                    failure_code="element_not_interactable",
                )
            )

        for located in cls._LOCATE_ACTION.finditer(message):
            target = cls._clean_target(located.group("target") or "")
            if (
                not target
                or re.search(
                    r"输入框|搜索框|textbox|input|field|box", target, re.IGNORECASE
                )
                or any(
                    item.kind in {"click", "seek"}
                    and abs(item.position - located.start()) < 8
                    for item in actions
                )
            ):
                continue
            actions.append(
                BrowserAction(
                    "seek",
                    located.start(),
                    semantic_target=target,
                    role="text",
                    success_condition="target_visible",
                    failure_code="target_not_found",
                )
            )

        typed = re.search(
            r"(?:(?:在|向)\s*(?P<target>[^，。！？;]{1,80}?)\s*)?"
            r"(?:逐字符)?(?:输入|写入|填入|\btype\b)\s*[：:]?\s*"
            r"[\"'“”`《》]?(?P<value>[A-Za-z0-9_@.+\- ]{1,500})[\"'“”`《》]?",
            message,
            re.IGNORECASE,
        )
        if typed:
            value = typed.group("value").strip()
            target = cls._clean_target(typed.group("target") or "输入框")
            if target in {"里面", "其中", "这里", "那里面", "it", "there"}:
                context = message[max(0, typed.start() - 100) : typed.start()]
                referenced = list(
                    re.finditer(
                        r"(?:搜索|search)\s*(?:输入框|框|field|box)|"
                        r"(?:输入框|textbox|field)",
                        context,
                        re.IGNORECASE,
                    )
                )
                if referenced:
                    target = referenced[-1].group(0)
            actions.append(
                BrowserAction(
                    "type",
                    typed.start(),
                    semantic_target=target,
                    value=value,
                    success_condition="input_value_equals_expected",
                    failure_code="input_state_mismatch",
                )
            )
        elif re.search(r"清空|\bclear\b", message, re.IGNORECASE):
            cleared = re.search(
                r"(?:清空|\bclear\b)\s*(?P<target>[^，。！？;\n]*)",
                message,
                re.IGNORECASE,
            )
            assert cleared is not None
            actions.append(
                BrowserAction(
                    "type",
                    cleared.start(),
                    semantic_target=cls._clean_target(
                        cleared.group("target") or "输入框"
                    ),
                    value="",
                    success_condition="input_value_equals_expected",
                    failure_code="input_state_mismatch",
                )
            )

        seek = re.search(
            r"(?:滚动|\bscroll\b).*?(?:直到|寻找|查找|找到|(?<![找看])到|\buntil\b|\bfind\b)\s*"
            r"(?:(?:页面里的|页面中|the\s+)?(?:标题|heading|章节|section|文本|text)\s*)?"
            r"(?:[\"'“`《](?P<quoted>[^\"'”`》\n]+)[\"'”`》]|"
            r"(?P<plain>[^，。！？；;\n]+))",
            message,
            re.IGNORECASE,
        )
        if seek:
            target = cls._clean_target(
                seek.group("quoted") or seek.group("plain") or ""
            )
            actions = [
                item
                for item in actions
                if not (
                    item.kind == "seek"
                    and (
                        item.semantic_target == target
                        or abs(item.position - seek.start()) < 40
                    )
                )
            ]
            actions.append(
                BrowserAction(
                    "seek",
                    seek.start(),
                    semantic_target=target,
                    role="heading"
                    if re.search(r"标题|heading", seek.group(0), re.IGNORECASE)
                    else "text",
                    success_condition="target_visible",
                    failure_code="target_not_found",
                )
            )
        elif re.search(r"滚动|\bscroll\b", message, re.IGNORECASE):
            matched = re.search(r"滚动|\bscroll\b", message, re.IGNORECASE)
            assert matched is not None
            actions.append(BrowserAction("scroll", matched.start()))
        elif cls._CONTINUATION_SCROLL.search(message):
            matched = cls._CONTINUATION_SCROLL.search(message)
            assert matched is not None
            actions.append(BrowserAction("scroll", matched.start()))

        declarative_actions: tuple[tuple[str, BrowserActionKind], ...] = (
            (r"截(?:个)?图|\bscreenshot\b", "screenshot"),
            (r"后退|返回上一页|\bbrowser\s+back\b|\bgo\s+back\b", "back"),
            (r"前进|\bgo\s+forward\b", "forward"),
            (r"刷新|重新加载|\breload\b|\brefresh\b", "refresh"),
        )
        for pattern, kind in declarative_actions:
            matched = re.search(pattern, message, re.IGNORECASE)
            if matched:
                actions.append(BrowserAction(kind, matched.start()))

        tabs = re.search(
            r"(?:检查|查看|列出|看看).{0,12}(?:tabs?|标签页)|(?:tabs?|标签页).{0,12}(?:情况|列表)",
            message,
            re.IGNORECASE,
        )
        if tabs:
            actions.append(
                BrowserAction(
                    "tabs", tabs.start(), success_condition="tab_list_observed"
                )
            )

        switch = re.search(
            r"(?:切换到|switch\s+to)\s*(?:第)?(?P<ordinal>[一二三四五\d]+)(?:个)?\s*(?:tab|标签页)",
            message,
            re.IGNORECASE,
        )
        if switch:
            actions.append(
                BrowserAction(
                    "switch_tab",
                    switch.start(),
                    ordinal=cls._ordinal(switch.group("ordinal")),
                )
            )
        close = re.search(
            r"(?:关闭|close)\s*(?:第)?(?P<ordinal>[一二三四五\d]+)(?:个)?\s*(?:tab|标签页)",
            message,
            re.IGNORECASE,
        )
        if close:
            actions.append(
                BrowserAction(
                    "close_tab",
                    close.start(),
                    ordinal=cls._ordinal(close.group("ordinal")),
                )
            )
        close_all = re.search(
            r"(?:关闭|close)\s*"
            r"(?:当前(?:浏览器|browser)(?:中|内|里)(?:的)?\s*)?"
            r"(?:所有|全部|all)\s*"
            r"(?:页面(?:\s*/\s*标签页)?|标签页|tabs?)",
            message,
            re.IGNORECASE,
        )
        if close_all and not close:
            actions.append(
                BrowserAction(
                    "close_tab",
                    close_all.start(),
                    semantic_target="all",
                    success_condition="all_listed_tabs_closed",
                )
            )
        if not destinations:
            blank_tab = cls._NEW_TAB.search(message)
            if blank_tab:
                actions.append(BrowserAction("new_tab", blank_tab.start()))

        read_actions = list(cls._READ_ACTION.finditer(message))
        if read_actions and not any(item.kind == "tabs" for item in actions):
            actions.extend(
                BrowserAction(
                    "read_page",
                    matched.start(),
                    success_condition="page_semantics_observed",
                    failure_code="page_read_failed",
                )
                for matched in read_actions
            )
        elif (
            not destinations
            and not actions
            and cls._READ.search(message)
            and cls._CURRENT.search(message)
        ):
            actions.append(
                BrowserAction(
                    "read_page", 0, success_condition="page_semantics_observed"
                )
            )

        actions.sort(key=lambda item: item.position)
        unresolved = (
            None
            if not destinations and cls._OPEN_ENDED.search(message)
            else cls._unresolved_destination_request(message)
        )
        requires_resolution = any(item.url is None for item in destinations)
        requires_resolution = requires_resolution or bool(
            unresolved and not destinations and not cls._CURRENT.search(message)
        )
        if requires_resolution and unresolved and not destinations:
            hint = re.split(
                r"(?:看看|看一下|然后|接着|随后|并且|\bthen\b)",
                unresolved.group("target"),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            destinations = (
                BrowserDestination(None, hint or None, "model", unresolved.start()),
            )
            actions = [
                item
                for item in actions
                if not (
                    item.kind == "click"
                    and unresolved.start() <= item.position < unresolved.end()
                )
            ]
            actions.append(
                BrowserAction(
                    "navigate",
                    unresolved.start(),
                    destination_index=0,
                    success_condition="final_url_observed",
                    failure_code="navigation_failed",
                )
            )
        actions.sort(key=lambda item: item.position)
        scope: Literal["current", "destination", "multiple"] = (
            "multiple"
            if len(destinations) > 1
            else "destination"
            if destinations
            else "current"
        )
        tab_policy: Literal["current", "new_tab", "mixed"] = (
            "mixed" if any(item.kind == "new_tab" for item in actions) else "current"
        )
        return BrowserActionPlan(
            destinations=destinations,
            active_page_scope=scope,
            actions=tuple(actions),
            tab_policy=tab_policy,
            requires_destination_resolution=requires_resolution,
        )

    @classmethod
    def build_intent(
        cls, message: str, *, source: IntentSource = "user"
    ) -> BrowserIntent:
        """Parse once into semantic intent, then project mechanically to execution.

        The source is context metadata. Parsing and execution requirements are the
        same for user, self, future runtime, and continuation origins.
        """

        constraints = cls._semantic_constraints(message)
        no_browser = any(item.kind == "prohibit_browser" for item in constraints)
        status_question = bool(cls._STATUS_QUESTION.search(message))
        current_request = bool(cls._CURRENT_REQUEST_STATUS.search(message))
        if status_question or (no_browser and current_request):
            return BrowserIntent(
                version="browser-intent/2b-v1",
                source=source,
                task_kind=(
                    "execution_status_query"
                    if current_request
                    else "historical_evidence_query"
                ),
                page_scope="contextual",
                historical_reference=(
                    "current_request" if current_request else "previous_request"
                ),
                constraints=constraints,
            )
        if no_browser:
            return BrowserIntent(
                version="browser-intent/2b-v1",
                source=source,
                task_kind="non_browser_response",
                constraints=constraints,
            )

        candidate = cls._build_candidate_plan(message)
        if (
            (
                cls._TAKEOVER_TASK.search(message)
                or cls._CAPABILITY_QUERY.search(message)
            )
            and not cls._URL.search(message)
            and not cls._DOMAIN.search(message)
        ):
            # Authentication/takeover routing and capability self-knowledge are
            # existing boundaries outside Phase 2B. Do not turn their nouns into
            # deterministic page actions.
            candidate = BrowserActionPlan()
        open_ended = (
            bool(cls._OPEN_ENDED.search(message)) and not candidate.destinations
        )
        actions = [
            action
            for action in candidate.actions
            if not cls._position_is_prohibited(
                action.position, action.kind, constraints
            )
        ]

        # A prohibition may use the broader word “search” while a mechanical page
        # operation would be represented as seek. It still must not be inverted.
        prohibited_actions = {item.action for item in constraints}
        if "search" in prohibited_actions:
            actions = [item for item in actions if item.kind != "seek"]

        page_scope: Literal[
            "current", "contextual", "destination", "multiple", "unspecified"
        ]
        if cls._CURRENT.search(message):
            page_scope = "contextual"
        elif len(candidate.destinations) > 1:
            page_scope = "multiple"
        elif candidate.destinations:
            page_scope = "destination"
        elif actions:
            page_scope = "current"
        else:
            page_scope = "unspecified"

        if open_ended:
            return BrowserIntent(
                version="browser-intent/2b-v1",
                source=source,
                task_kind="open_ended_browse",
                page_scope=page_scope,
                constraints=constraints,
                ambiguity=candidate.ambiguity,
            )

        stable_actions = cls._with_stable_subgoal_ids(actions)
        task_kind: BrowserTaskKind = (
            "browser_task"
            if stable_actions or candidate.requires_destination_resolution
            else "none"
        )
        return BrowserIntent(
            version="browser-intent/2b-v1",
            source=source,
            task_kind=task_kind,
            destinations=candidate.destinations,
            page_scope=page_scope,
            constraints=constraints,
            required_subgoals=stable_actions,
            ambiguity=candidate.ambiguity,
            requires_destination_resolution=candidate.requires_destination_resolution,
        )

    @classmethod
    def build(cls, message: str) -> BrowserActionPlan:
        """Compatibility projection for existing execution code and callers."""

        return cls.build_intent(message).to_action_plan()
