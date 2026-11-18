"""输出安全节点：Graph结束前对最终AI回复做合规检查

按风险等级分两级处理（均为确定性规则，不额外调用模型、不增加延迟）：
1. 中危——原位改写后放行：
   - PII泄露：手机号/身份证号/银行卡号 → 掩码脱敏
   - 违规话术：承诺收益等保险销售禁词 → 改写为合规表述，并追加风险提示
2. 高危——替换整条回复并转人工（human_handoff）：
   教唆隐瞒健康告知、诱导退保换新等，违规原文不落库

注意：流式对话中token已实时推送给前端，本节点保证的是落库历史与后续
轮次引用的合规；生成侧的约束由各子Agent的SystemPrompt承担。
"""

import re
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.constants import END
from langgraph.types import Command

from app.core.logging import get_logger
from .schemas import InsuranceAgentState

logger = get_logger(__name__)


# 1.PII脱敏规则（按顺序匹配，前后不能紧邻数字/掩码符，避免重复掩码）
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 身份证号（18位，末位可为X），需先于银行卡号匹配
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    # 手机号
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    # 银行卡号（13~19位数字）
    ("bank_card", re.compile(r"(?<![\d*])\d{13,19}(?![\d*])")),
]

# 2.违规话术改写规则：保险销售禁词 → 合规表述
VIOLATION_REPLACEMENTS: dict[str, str] = {
    "保本保息": "不承诺保本保息",
    "稳赚不赔": "收益存在不确定性",
    "保证收益": "收益存在不确定性",
    "百分百赔付": "按保险合同条款赔付",
    "100%赔付": "按保险合同条款赔付",
    "绝对安全": "相对稳健",
    "没有任何风险": "存在一定风险",
    "无风险": "风险较低",
}

# 拼成单个正则，长词在前，保证"没有任何风险"优先于"无风险"命中
VIOLATION_PATTERN = re.compile(
    "|".join(sorted(map(re.escape, VIOLATION_REPLACEMENTS), key=len, reverse=True))
)

# 命中违规话术时追加的固定风险提示
RISK_DISCLAIMER = "\n\n（温馨提示：保险产品的具体权益与赔付条件以保险合同条款为准，请勿仅凭本回复做出投保决定。）"

# 3.高危规则：命中即不做原位改写，替换整条回复并转人工
SEVERE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 教唆隐瞒健康告知，可能导致保单无效
    ("conceal_disclosure", re.compile(r"(不用|无需|别)做?健康告知|隐瞒(病史|既往症|体检异常)")),
    # 诱导退旧保新
    ("induce_lapse", re.compile(r"退保.{0,12}(更划算|更合适|买新的)")),
    # 协助规避核保/理赔审核
    ("evade_review", re.compile(r"(规避|绕过|应付)(核保|理赔)(审核|调查)?")),
    # 骗保倾向
    ("fraud", re.compile(r"骗保|带病投保.{0,6}(没事|可以|能赔)")),
]

# 高危命中后替换原回复的固定话术，违规原文不落库
SEVERE_NOTICE = "抱歉，刚才的回复涉及需要人工核实的内容，已为您转接人工客服，请稍候。"


def find_severe_hits(text: str) -> list[str]:
    """返回命中的高危规则名列表"""

    return [name for name, pattern in SEVERE_PATTERNS if pattern.search(text)]


def _mask_digits(match: re.Match) -> str:
    """数字掩码：保留前3后4，中间打星"""

    digits = match.group(0)
    return f"{digits[:3]}{'*' * (len(digits) - 7)}{digits[-4:]}"


def sanitize_text(text: str) -> tuple[str, list[str]]:
    """对文本做PII脱敏与违规话术改写，返回（新文本, 命中的规则列表）"""

    hits: list[str] = []
    sanitized = text

    # 1.PII脱敏
    for name, pattern in PII_PATTERNS:
        if pattern.search(sanitized):
            sanitized = pattern.sub(_mask_digits, sanitized)
            hits.append(name)

    # 2.违规话术改写，命中则追加一次风险提示
    violations = set(VIOLATION_PATTERN.findall(sanitized))
    if violations:
        sanitized = VIOLATION_PATTERN.sub(
            lambda match: VIOLATION_REPLACEMENTS[match.group(0)], sanitized
        )
        sanitized += RISK_DISCLAIMER
        hits.extend(sorted(violations))

    return sanitized, hits


def find_last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    """查找最近一条可展示的AI回复（跳过带工具调用的消息）"""

    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls and message.text:
            return message
    return None


async def safety_node(state: InsuranceAgentState) -> Command[Literal["human_handoff", "__end__"]]:
    """安全节点：中危原位改写后放行，高危替换回复并转人工"""

    last = find_last_ai_message(state["messages"])
    if last is None:
        return Command(goto=END)

    # 1.高危：替换整条回复（按消息id覆盖，违规原文不落库）并转人工
    severe_hits = find_severe_hits(last.text)
    if severe_hits:
        logger.warning("安全节点高危拦截，转人工", message_id=last.id, rules=severe_hits)
        return Command(
            update={
                "active_workflow": "human_handoff",
                "messages": [AIMessage(content=SEVERE_NOTICE, id=last.id)],
            },
            goto="human_handoff",
        )

    # 2.中危：原位改写后放行
    sanitized, hits = sanitize_text(last.text)
    if not hits:
        return Command(goto=END)

    logger.warning("安全节点拦截", message_id=last.id, rules=hits)
    return Command(
        update={"messages": [AIMessage(content=sanitized, id=last.id)]},
        goto=END,
    )
