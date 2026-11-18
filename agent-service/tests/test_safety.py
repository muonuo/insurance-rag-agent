"""app.agents.safety 输出安全节点测试：PII脱敏、违规话术改写与节点路由行为"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.constants import END

from app.agents.safety import (
    RISK_DISCLAIMER,
    SEVERE_NOTICE,
    find_severe_hits,
    sanitize_text,
    safety_node,
)


class TestSanitizePII:
    def test_phone_masked(self):
        sanitized, hits = sanitize_text("您可以回拨13812345678联系我")
        assert "13812345678" not in sanitized
        assert "138****5678" in sanitized
        assert hits == ["phone"]

    def test_id_card_masked(self):
        sanitized, hits = sanitize_text("已收到身份证号110101199001011234")
        assert "110101199001011234" not in sanitized
        assert "110***********1234" in sanitized
        assert hits == ["id_card"]

    def test_bank_card_masked(self):
        # 19位数字不会先被身份证规则截断匹配
        sanitized, hits = sanitize_text("卡号6222020200112233445已登记")
        assert "6222020200112233445" not in sanitized
        # 掩码保留前3后4，中间补星
        assert "622" + "*" * 12 + "3445" in sanitized
        assert hits == ["bank_card"]


class TestSanitizeViolation:
    def test_guarantee_phrases_rewritten_with_disclaimer(self):
        sanitized, hits = sanitize_text("这款产品保证收益，稳赚不赔。")
        assert "保证收益" not in sanitized
        assert "稳赚不赔" not in sanitized
        assert "收益存在不确定性" in sanitized
        assert RISK_DISCLAIMER in sanitized
        assert set(hits) == {"保证收益", "稳赚不赔"}

    def test_longer_phrase_takes_priority(self):
        # "没有任何风险"应整体命中，而不是先被"无风险"替换
        sanitized, hits = sanitize_text("这份保险没有任何风险")
        assert "存在一定风险" in sanitized
        assert "无风险" not in sanitized
        assert hits == ["没有任何风险"]

    def test_clean_text_unchanged(self):
        text = "具体保障范围以保险合同条款为准。"
        sanitized, hits = sanitize_text(text)
        assert sanitized == text
        assert hits == []


class TestFindSevereHits:
    def test_conceal_disclosure(self):
        assert find_severe_hits("放心，不用健康告知，直接投保就行") == ["conceal_disclosure"]

    def test_induce_lapse(self):
        assert find_severe_hits("建议您把旧保单退保，买新的更划算") == ["induce_lapse"]

    def test_moderate_violation_is_not_severe(self):
        # 中危话术不触发转人工
        assert find_severe_hits("这款产品保证收益") == []


class TestSafetyNode:
    def test_severe_hit_routes_to_human_handoff(self):
        state = {
            "messages": [
                HumanMessage("能不告知病史吗"),
                AIMessage("放心，不用健康告知，直接投保就行", id="ai-1"),
            ]
        }
        result = asyncio.run(safety_node(state))
        assert result.goto == "human_handoff"
        assert result.update["active_workflow"] == "human_handoff"
        (message,) = result.update["messages"]
        # 相同id覆盖，违规原文不落库
        assert message.id == "ai-1"
        assert message.text == SEVERE_NOTICE

    def test_severe_takes_priority_over_moderate(self):
        # 同时命中高危与中危时，按高危转人工处理
        state = {
            "messages": [
                HumanMessage("这个产品好吗"),
                AIMessage("不用健康告知，而且保证收益", id="ai-2"),
            ]
        }
        result = asyncio.run(safety_node(state))
        assert result.goto == "human_handoff"

    def test_moderate_hit_rewrites_and_ends(self):
        state = {
            "messages": [
                HumanMessage("这个产品有风险吗"),
                AIMessage("放心，这款产品无风险", id="ai-3"),
            ]
        }
        result = asyncio.run(safety_node(state))
        assert result.goto == END
        (message,) = result.update["messages"]
        assert message.id == "ai-3"
        assert "无风险" not in message.text
        assert "风险较低" in message.text

    def test_skips_tool_call_messages(self):
        # 带工具调用的AIMessage不是最终回复，应跳过后找到真正的正文
        state = {
            "messages": [
                AIMessage("", tool_calls=[{"name": "search", "args": {}, "id": "t-1"}]),
                AIMessage("本产品保证收益很高", id="ai-4"),
            ]
        }
        result = asyncio.run(safety_node(state))
        assert result.goto == END
        (message,) = result.update["messages"]
        assert message.id == "ai-4"
        assert "保证收益" not in message.text

    def test_clean_message_passes_through(self):
        state = {
            "messages": [
                HumanMessage("你好"),
                AIMessage("您好，请问需要什么帮助？", id="ai-5"),
            ]
        }
        result = asyncio.run(safety_node(state))
        assert result.goto == END
        assert not result.update

    def test_human_handoff_reply_passes_safety(self):
        # 转人工后流程会回到safety收尾，其固定回复必须能通过检查，
        # 否则human_handoff与safety之间会陷入死循环
        state = {
            "messages": [
                HumanMessage("我要投诉"),
                AIMessage("你的请求已转人工，请耐心等待~~", id="ai-6"),
            ]
        }
        result = asyncio.run(safety_node(state))
        assert result.goto == END
        assert not result.update
