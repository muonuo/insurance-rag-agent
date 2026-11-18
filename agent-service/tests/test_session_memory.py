"""会话短期记忆测试：窗口截断、摘要增量更新、fail-open 与中间件压缩"""

import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents.memory.middleware import SessionMemoryMiddleware
from app.agents.memory.session_memory import (
    SessionMemory,
    SessionSnapshot,
    split_history_and_current,
)


class FakeRedis:
    """内存版Redis：实现mget/setex/delete"""

    def __init__(self, fail: bool = False):
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.fail = fail

    def _check(self):
        if self.fail:
            raise RuntimeError("redis down")

    async def mget(self, *keys):
        self._check()
        return [self.data.get(k) for k in keys]

    async def setex(self, key, ttl, value):
        self._check()
        self.data[key] = value
        self.ttls[key] = ttl

    async def delete(self, *keys):
        self._check()
        for k in keys:
            self.data.pop(k, None)
            self.ttls.pop(k, None)


def make_memory(client, enabled=True, recent_limit=10) -> SessionMemory:
    memory = SessionMemory(
        url="redis://fake/1", ttl=3600, recent_limit=recent_limit, enabled=enabled
    )
    memory._client = client
    return memory


def dialog(n: int) -> list:
    """构造 n 轮人工/助手消息"""

    msgs = []
    for i in range(n):
        msgs.append(HumanMessage(f"问题{i}"))
        msgs.append(AIMessage(f"回答{i}"))
    return msgs


class TestSplit:
    def test_last_human_is_boundary(self):
        msgs = dialog(2) + [
            HumanMessage("本轮问题"),
            AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
            ToolMessage("工具结果", tool_call_id="1"),
        ]
        history, current = split_history_and_current(msgs)

        assert len(history) == 4
        # 当前轮包含用户消息和工具调用链，必须原样保留
        assert current[0].text == "本轮问题"
        assert isinstance(current[2], ToolMessage)

    def test_no_human_means_all_current(self):
        history, current = split_history_and_current([AIMessage("开场")])
        assert history == []
        assert len(current) == 1


class TestReadWrite:
    def test_miss_returns_none(self):
        memory = make_memory(FakeRedis())
        assert asyncio.run(memory.get("t1")) is None

    def test_set_then_get_roundtrip(self):
        client = FakeRedis()
        memory = make_memory(client)

        asyncio.run(memory.set("t1", "旧摘要", [{"role": "human", "content": "你好"}], 4))
        snap = asyncio.run(memory.get("t1"))

        assert snap.summary == "旧摘要"
        assert snap.recent[0]["content"] == "你好"
        assert snap.summarized_count == 4
        # TTL 写入
        assert all(ttl == 3600 for ttl in client.ttls.values())

    def test_append_turn_trims_window(self):
        memory = make_memory(FakeRedis(), recent_limit=4)
        asyncio.run(memory.set("t1", None, [], 0))

        for i in range(3):
            asyncio.run(memory.append_turn("t1", f"问{i}", f"答{i}"))

        snap = asyncio.run(memory.get("t1"))
        assert len(snap.recent) == 4
        assert snap.recent[-2]["content"] == "问2"
        assert snap.recent[-1]["content"] == "答2"

    def test_append_turn_skips_empty(self):
        client = FakeRedis()
        memory = make_memory(client)
        asyncio.run(memory.append_turn("t1", "", "答"))
        assert client.data == {}

    def test_invalidate_deletes_keys(self):
        client = FakeRedis()
        memory = make_memory(client)
        asyncio.run(memory.set("t1", "摘要", [], 0))

        asyncio.run(memory.invalidate("t1"))
        assert client.data == {}


class TestFailOpen:
    def test_get_failure_returns_none(self):
        memory = make_memory(FakeRedis(fail=True))
        assert asyncio.run(memory.get("t1")) is None

    def test_write_failures_do_not_raise(self):
        memory = make_memory(FakeRedis(fail=True))
        asyncio.run(memory.set("t1", "s", [], 0))
        asyncio.run(memory.append_turn("t1", "问", "答"))
        asyncio.run(memory.invalidate("t1"))


class TestRefresh:
    def _patch_summarize(self, memory, calls, result="合并后的摘要"):
        async def fake_summarize(previous, messages):
            calls.append((previous, [m.text for m in messages]))
            return result

        memory.summarize = fake_summarize  # type: ignore[method-assign]

    def test_builds_window_and_summary(self):
        memory = make_memory(FakeRedis(), recent_limit=4)
        calls: list = []
        self._patch_summarize(memory, calls)

        asyncio.run(memory.refresh("t1", dialog(5)))  # 10条消息

        snap = asyncio.run(memory.get("t1"))
        assert len(snap.recent) == 4
        assert snap.summary == "合并后的摘要"
        assert snap.summarized_count == 6
        # 摘要只收到溢出的6条
        assert len(calls[0][1]) == 6

    def test_incremental_summary_only_new_part(self):
        memory = make_memory(FakeRedis(), recent_limit=4)
        calls: list = []
        self._patch_summarize(memory, calls)

        asyncio.run(memory.refresh("t1", dialog(4)))   # 8条 → 溢出4条
        asyncio.run(memory.refresh("t1", dialog(6)))   # 12条 → 溢出8条，只该摘要新增4条

        assert len(calls) == 2
        assert len(calls[1][1]) == 4
        # 第二次把上一次的摘要作为 previous 传入
        assert calls[1][0] == "合并后的摘要"

    def test_summary_failure_keeps_previous(self):
        memory = make_memory(FakeRedis(), recent_limit=4)
        asyncio.run(memory.set("t1", "旧摘要", [], 0))

        async def boom(previous, messages):
            raise RuntimeError("model down")

        memory.summarize = boom  # type: ignore[method-assign]
        asyncio.run(memory.refresh("t1", dialog(5)))

        snap = asyncio.run(memory.get("t1"))
        assert snap.summary == "旧摘要"


class FakeRequest:
    def __init__(self, messages, context):
        self.messages = messages
        self.runtime = SimpleNamespace(context=context)

    def override(self, **kwargs):
        return FakeRequest(kwargs.get("messages", self.messages), self.runtime.context)


class FakeMemory:
    def __init__(self, snapshot=None, enabled=True):
        self.snapshot = snapshot
        self.enabled = enabled
        self.recent_limit = 4
        self.refreshed: list = []

    async def get(self, thread_id):
        return self.snapshot

    def maybe_schedule_refresh(self, thread_id, history, snapshot):
        self.refreshed.append(thread_id)


class TestMiddleware:
    def _run(self, memory, messages, context):
        middleware = SessionMemoryMiddleware(memory)
        captured = {}

        async def handler(req):
            captured["messages"] = req.messages
            return "ok"

        asyncio.run(
            middleware.awrap_model_call(FakeRequest(messages, context), handler)
        )
        return captured["messages"]

    def test_compacts_history_with_summary_and_recent(self):
        snapshot = SessionSnapshot(
            summary="用户预算3000元",
            recent=[
                {"role": "human", "content": "问题9"},
                {"role": "ai", "content": "回答9"},
            ],
            summarized_count=8,
        )
        memory = FakeMemory(snapshot)
        history = dialog(6)  # 12条历史
        current = [HumanMessage("本轮")]

        result = self._run(
            memory, history + current, SimpleNamespace(thread_id="t1")
        )

        assert isinstance(result[0], SystemMessage)
        assert "用户预算3000元" in result[0].text
        assert [m.text for m in result[1:3]] == ["问题9", "回答9"]
        assert result[-1].text == "本轮"
        # 触发了后台刷新
        assert memory.refreshed == ["t1"]

    def test_cache_miss_falls_back_to_window(self):
        memory = FakeMemory(None)
        history = dialog(6)
        current = [HumanMessage("本轮")]

        result = self._run(
            memory, history + current, SimpleNamespace(thread_id="t1")
        )

        # 无摘要，取最近4条历史 + 当前轮
        assert len(result) == 5
        assert not any(isinstance(m, SystemMessage) for m in result)
        assert result[-1].text == "本轮"

    def test_no_thread_id_passthrough(self):
        memory = FakeMemory(None)
        messages = dialog(3)

        result = self._run(memory, messages, SimpleNamespace(thread_id=None))

        assert result == messages
        assert memory.refreshed == []

    def test_disabled_passthrough(self):
        memory = FakeMemory(None, enabled=False)
        messages = dialog(3)

        result = self._run(memory, messages, SimpleNamespace(thread_id="t1"))

        assert result == messages
