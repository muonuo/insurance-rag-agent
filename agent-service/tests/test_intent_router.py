"""app.agents.intent_router 测试：多轮历史窗口构造 + 画像个性化问候"""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.intent_router import (
    GREETING_SYSTEM_PROMPT,
    ROUTE_CONTEXT_MAX_CHARS,
    ROUTE_CONTEXT_MESSAGES,
    SOCIAL_RESPONSES,
    build_route_context,
    router,
)
from app.agents.profile import UserProfile, summarize_profile


class TestBuildRouteContext:
    def test_includes_workflow_and_recent_turns(self):
        messages = [
            HumanMessage("我想买保险"),
            AIMessage("好的，请问您的预算是多少？"),
            HumanMessage("预算3000"),
        ]
        context = build_route_context(messages, "recommendation_plan")

        assert "上一轮工作流：recommendation_plan" in context
        assert "用户: 我想买保险" in context
        assert "AI: 好的，请问您的预算是多少？" in context

    def test_excludes_current_user_message(self):
        # 本轮提问作为query单独传给路由，上下文里不应重复出现
        messages = [HumanMessage("第一轮"), AIMessage("回复"), HumanMessage("本轮提问")]
        context = build_route_context(messages, "")
        assert "本轮提问" not in context

    def test_skips_tool_messages_and_tool_calls(self):
        # 工具调用等中间过程不进入路由上下文
        messages = [
            HumanMessage("查下理赔进度"),
            AIMessage("", tool_calls=[{"name": "query_user_claims", "args": {}, "id": "t-1"}]),
            ToolMessage("raw tool result", tool_call_id="t-1"),
            AIMessage("您的理赔在审核中"),
            HumanMessage("那材料还缺什么"),
        ]
        context = build_route_context(messages, "claim")
        assert "query_user_claims" not in context
        assert "raw tool result" not in context
        assert "AI: 您的理赔在审核中" in context

    def test_window_limit_keeps_recent_only(self):
        # 超出窗口的旧消息不进入上下文
        messages: list = []
        for i in range(ROUTE_CONTEXT_MESSAGES + 4):
            messages.append(HumanMessage(f"消息{i}"))
            messages.append(AIMessage(f"回复{i}"))
        messages.append(HumanMessage("本轮"))

        context = build_route_context(messages, "")
        assert "消息0" not in context
        # 窗口内保留最近的消息
        assert "用户: 消息7" in context
        assert "AI: 回复9" in context

    def test_truncates_long_messages(self):
        long_text = "长" * (ROUTE_CONTEXT_MAX_CHARS + 300)
        context = build_route_context(
            [HumanMessage(long_text), HumanMessage("本轮")], ""
        )
        user_line = next(line for line in context.splitlines() if line.startswith("用户:"))
        assert len(user_line) == len("用户: ") + ROUTE_CONTEXT_MAX_CHARS

    def test_empty_history_returns_workflow_only(self):
        context = build_route_context([HumanMessage("你好")], "")
        assert context == "上一轮工作流：无"


class FakeGreetModel:
    """假问候模型：记录收到的提示词，返回预设文本"""

    def __init__(self, text: str = "王先生您好！上次说预算每年8000，重疾险考虑得怎么样啦？"):
        self.text = text
        self.calls: list[str] = []

    async def ainvoke(self, messages):
        self.calls.append(str(messages[-1].content))
        return SimpleNamespace(text=self.text)


class FailingGreetModel:
    async def ainvoke(self, messages):
        raise RuntimeError("network down")


def make_profile() -> UserProfile:
    return UserProfile(age=30, family="已婚，有一个3岁女儿", budget="每年8000元以内")


class TestSummarizeProfile:
    def test_none_returns_none(self):
        assert summarize_profile(None) is None

    def test_empty_profile_returns_none(self):
        assert summarize_profile(UserProfile()) is None

    def test_fields_summarized_in_order(self):
        summary = summarize_profile(make_profile())
        assert summary == "年龄：30；家庭：已婚，有一个3岁女儿；预算：每年8000元以内"

    def test_list_field_joined(self):
        profile = UserProfile(existing_coverage=["百万医疗险", "意外险"])
        assert summarize_profile(profile) == "已有保障：百万医疗险、意外险"


class TestGreetingPromptGuard:
    """问候提示词必须包含防幻觉约束：编造称呼/历史是线上实际出现过的bad-case"""

    def test_forbids_inventing_salutation(self):
        assert "不要编造称呼或姓氏" in GREETING_SYSTEM_PROMPT

    def test_forbids_inventing_history(self):
        assert "不要编造" in GREETING_SYSTEM_PROMPT
        assert "你之前说过" in GREETING_SYSTEM_PROMPT

    def test_limits_to_profile_facts(self):
        assert "只能使用给定画像中的事实" in GREETING_SYSTEM_PROMPT


class TestPersonalGreeting:
    def test_greeting_with_profile_is_personalized(self, monkeypatch):
        fake = FakeGreetModel()
        monkeypatch.setattr(router, "greet_model", fake)

        result = asyncio.run(router.route("你好", "上一轮工作流：无", make_profile()))

        assert result == fake.text
        # 提示词里带上了画像摘要和用户原话
        assert "每年8000元以内" in fake.calls[0]
        assert "你好" in fake.calls[0]

    def test_greeting_without_profile_stays_canned(self, monkeypatch):
        fake = FakeGreetModel()
        monkeypatch.setattr(router, "greet_model", fake)

        result = asyncio.run(router.route("你好", "", None))

        assert result == SOCIAL_RESPONSES["greeting"]
        # 没画像不花模型钱
        assert fake.calls == []

    def test_greeting_with_empty_profile_stays_canned(self, monkeypatch):
        fake = FakeGreetModel()
        monkeypatch.setattr(router, "greet_model", fake)

        result = asyncio.run(router.route("你好", "", UserProfile()))

        assert result == SOCIAL_RESPONSES["greeting"]
        assert fake.calls == []

    def test_greeting_model_failure_falls_back(self, monkeypatch):
        monkeypatch.setattr(router, "greet_model", FailingGreetModel())

        result = asyncio.run(router.route("你好", "", make_profile()))

        # 模型挂了也不能让寒暄失败
        assert result == SOCIAL_RESPONSES["greeting"]

    def test_thanks_with_profile_stays_canned(self, monkeypatch):
        # 结束类寒暄不做个性化，避免答非所问的追问
        fake = FakeGreetModel()
        monkeypatch.setattr(router, "greet_model", fake)

        result = asyncio.run(router.route("谢谢", "", make_profile()))

        assert result == SOCIAL_RESPONSES["thanks"]
        assert fake.calls == []
