"""ChatThreadService 历史会话隔离测试：验证不同用户只能取到自己的会话历史"""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.modules.chat_thread.exceptions import ChatThreadNotFound
from app.modules.chat_thread.service import ChatThreadService


class FakeChatThreadRepository:
    """内存版会话仓库，required_owner语义与真实SQL一致（id+user_id双条件）"""

    def __init__(self, threads: list[SimpleNamespace]):
        self.threads = threads

    async def required_owner(self, user_id: int, thread_id: str):
        return next(
            (
                t
                for t in self.threads
                if t.user_id == user_id and str(t.id) == thread_id
            ),
            None,
        )


class FakeSession:
    """get_chat_history不需要事务，session不会被使用"""

    pass


class FakeAgent:
    """假CompiledStateGraph：模拟checkpointer按thread_id存取互不相通的快照"""

    def __init__(self, states: dict[str, dict]):
        self.states = states
        # 记录取快照的thread_id，用于验证越权时根本不会碰状态数据
        self.calls: list[str] = []

    async def aget_state(self, config: dict) -> SimpleNamespace:
        thread_id = config["configurable"]["thread_id"]
        self.calls.append(thread_id)
        values = self.states.get(thread_id, {})
        return SimpleNamespace(values=values, interrupts=None, tasks=())


def make_thread(user_id: int) -> SimpleNamespace:
    return SimpleNamespace(id=uuid4(), user_id=user_id, title=f"用户{user_id}的会话")


def make_state(question: str, answer: str) -> dict:
    return {
        "messages": [
            HumanMessage(question),
            AIMessage(answer, id="ai-1"),
        ],
        "additional_info": {"ai-1": {"sources": []}},
    }


def make_service(
    threads: list[SimpleNamespace], agent: FakeAgent
) -> ChatThreadService:
    service = ChatThreadService(session=FakeSession(), agent=agent)
    # 替换为内存仓库，隔离数据库依赖
    service.repository = FakeChatThreadRepository(threads)
    return service


class TestGetChatHistoryUserIsolation:
    def test_owner_reads_own_history(self):
        thread = make_thread(user_id=1)
        service = make_service([thread], FakeAgent({str(thread.id): make_state("A的问题", "A的回答")}))

        history = asyncio.run(service.get_chat_history(user_id=1, thread_id=str(thread.id)))

        assert [(m.role, m.content) for m in history.messages] == [
            ("user", "A的问题"),
            ("assistant", "A的回答"),
        ]
        # AI消息的引用信息按消息id正确关联
        assert history.messages[1].additional_info == {"sources": []}

    def test_other_user_gets_404_and_never_touches_state(self):
        # 用户2拿用户1的thread_id查历史 → 校验失败直接404
        thread = make_thread(user_id=1)
        agent = FakeAgent({str(thread.id): make_state("A的问题", "A的回答")})
        service = make_service([thread], agent)

        with pytest.raises(ChatThreadNotFound):
            asyncio.run(service.get_chat_history(user_id=2, thread_id=str(thread.id)))

        # 关键保证：越权请求连状态快照都没有读（在required_owner就拦下了）
        assert agent.calls == []

    def test_different_users_get_different_histories(self):
        # 两个用户各自会话的历史互不相同
        thread_1 = make_thread(user_id=1)
        thread_2 = make_thread(user_id=2)
        service = make_service(
            [thread_1, thread_2],
            FakeAgent(
                {
                    str(thread_1.id): make_state("重疾险怎么选", "建议关注保额和缴费期"),
                    str(thread_2.id): make_state("理赔进度", "您的理赔已在审核中"),
                }
            ),
        )

        history_1 = asyncio.run(service.get_chat_history(user_id=1, thread_id=str(thread_1.id)))
        history_2 = asyncio.run(service.get_chat_history(user_id=2, thread_id=str(thread_2.id)))

        assert [m.content for m in history_1.messages] == ["重疾险怎么选", "建议关注保额和缴费期"]
        assert [m.content for m in history_2.messages] == ["理赔进度", "您的理赔已在审核中"]

    def test_unknown_thread_gets_404(self):
        # 会话不存在时同样返回404，不泄露会话是否存在
        service = make_service([], FakeAgent({}))

        with pytest.raises(ChatThreadNotFound):
            asyncio.run(service.get_chat_history(user_id=1, thread_id=str(uuid4())))
