"""ChatService 在线评测接线测试：上下文捕获、回答累积、trace_id传递与失败隔离"""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import app.modules.chat.service as chat_service_module
from langchain_core.messages import AIMessage

from app.modules.chat.schema import ChatRequest
from app.modules.chat.service import ChatService, _AnswerBuffer
from app.rag.capture import record_contexts


class FakeStream:
    """假事件流：迭代过程中模拟工具调用写入检索上下文"""

    def __init__(self, events, interrupts=None, contexts=None):
        self.events = events
        self._interrupts = interrupts or []
        self._contexts = contexts or []

    async def __aiter__(self):
        for context in self._contexts:
            record_contexts([context])
        for event in self.events:
            yield event

    async def interrupts(self):
        return self._interrupts


class FakeAgent:
    def __init__(self, stream, active_workflow="chitchat"):
        self.stream = stream
        self.updated: list[dict] = []
        self.active_workflow = active_workflow

    async def astream_events(self, *args, **kwargs):
        return self.stream

    async def aupdate_state(self, config, value):
        self.updated.append(value)

    async def aget_state(self, config):
        return SimpleNamespace(values={"active_workflow": self.active_workflow})


class FakeRepository:
    async def required_owner(self, user_id, thread_id):
        return SimpleNamespace(id=thread_id, user_id=user_id)


def make_service(events, *, contexts=None, interrupts=None, active_workflow="chitchat"):
    agent = FakeAgent(
        FakeStream(events, interrupts=interrupts, contexts=contexts),
        active_workflow=active_workflow,
    )
    service = ChatService(session=None, agent=agent)
    service.repository = FakeRepository()
    return service, agent


def deltas(*texts):
    """构造 message-start + 增量 + message-finish 事件序列"""
    events = [{"method": "messages", "params": {"data": [{"event": "message-start", "id": "m1"}]}}]
    events += [
        {"method": "messages", "params": {"data": [{"delta": {"text": text}}]}}
        for text in texts
    ]
    events.append(
        {
            "method": "messages",
            "params": {
                "data": [
                    {
                        "event": "message-finish",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    }
                ]
            },
        }
    )
    return events


async def consume(service, request, user_id=1):
    return [item async for item in service.chat_stream(user_id, request)]


class TestAnswerBuffer:
    def test_deltas_joined_by_message_id(self):
        buffer = _AnswerBuffer()
        buffer.start("m1")
        buffer.add_delta("你")
        buffer.add_delta("好")
        buffer.finish()
        assert buffer.final_text("m1") == "你好"

    def test_whole_message_overrides_same_id_deltas(self):
        buffer = _AnswerBuffer()
        buffer.start("m1")
        buffer.add_delta("你")
        buffer.add_whole(AIMessage(content="你好", id="m1"))
        assert buffer.final_text("m1") == "你好"

    def test_orphan_deltas_fallback(self):
        buffer = _AnswerBuffer()
        buffer.add_delta("孤立")
        assert buffer.final_text(None) == "孤立"

    def test_empty(self):
        assert _AnswerBuffer().final_text(None) == ""


class TestChatStreamEvalWiring:
    def test_schedules_with_contexts_answer_and_trace_id(self, monkeypatch):
        scheduled: list[dict] = []
        monkeypatch.setattr(chat_service_module, "create_langfuse_handler", lambda: SimpleNamespace(last_trace_id="trace-1"))
        monkeypatch.setattr(
            chat_service_module,
            "maybe_schedule_eval",
            lambda **kwargs: scheduled.append(kwargs),
        )

        service, agent = make_service(
            deltas("免赔额", "是1万元"),
            contexts=["条款A", "条款A"],
        )
        request = ChatRequest(thread_id=uuid4(), message="免赔额多少")

        events = asyncio.run(consume(service, request))

        # SSE 正常结束
        assert events[-1].event == "done"
        assert len(scheduled) == 1
        call = scheduled[0]
        # 上下文按顺序去重
        assert call["contexts"] == ["条款A"]
        assert call["answer"] == "免赔额是1万元"
        assert call["trace_id"] == "trace-1"
        assert call["user_input"] == "免赔额多少"
        assert call["has_interrupt"] is False
        assert call["is_decision"] is False

    def test_schedule_exception_does_not_break_done(self, monkeypatch):
        monkeypatch.setattr(chat_service_module, "create_langfuse_handler", lambda: None)

        def boom(**kwargs):
            raise RuntimeError("scheduler down")

        monkeypatch.setattr(chat_service_module, "maybe_schedule_eval", boom)

        service, _ = make_service(deltas("回复"))
        request = ChatRequest(thread_id=uuid4(), message="问题")

        events = asyncio.run(consume(service, request))
        assert events[-1].event == "done"

    def test_interrupt_turn_passes_flag(self, monkeypatch):
        scheduled: list[dict] = []
        monkeypatch.setattr(chat_service_module, "create_langfuse_handler", lambda: SimpleNamespace(last_trace_id="trace-1"))
        monkeypatch.setattr(
            chat_service_module,
            "maybe_schedule_eval",
            lambda **kwargs: scheduled.append(kwargs),
        )

        service, _ = make_service(
            deltas("请确认方案"),
            interrupts=[SimpleNamespace(value={"plan": {}})],
        )
        request = ChatRequest(thread_id=uuid4(), message="推荐方案")

        asyncio.run(consume(service, request))
        assert scheduled[0]["has_interrupt"] is True

    def test_decision_turn_passes_flag(self, monkeypatch):
        scheduled: list[dict] = []
        monkeypatch.setattr(chat_service_module, "create_langfuse_handler", lambda: None)
        monkeypatch.setattr(
            chat_service_module,
            "maybe_schedule_eval",
            lambda **kwargs: scheduled.append(kwargs),
        )

        service, _ = make_service(deltas("已保存"))
        request = ChatRequest(thread_id=uuid4(), decision={"action": "approve"})

        asyncio.run(consume(service, request))
        assert scheduled[0]["is_decision"] is True

    def test_no_handler_means_no_trace_id(self, monkeypatch):
        scheduled: list[dict] = []
        monkeypatch.setattr(chat_service_module, "create_langfuse_handler", lambda: None)
        monkeypatch.setattr(
            chat_service_module,
            "maybe_schedule_eval",
            lambda **kwargs: scheduled.append(kwargs),
        )

        service, _ = make_service(deltas("回复"), contexts=["条款"])
        request = ChatRequest(thread_id=uuid4(), message="问题")

        asyncio.run(consume(service, request))
        # 交给 maybe_schedule_eval 的守卫去跳过
        assert scheduled[0]["trace_id"] is None


class TestWholeMessageMeta:
    """完整AIMessage（寒暄/个性化问候）也要能下发meta事件供前端反馈"""

    def whole_message_events(self):
        # 寒暄路径：route_node直接返回完整AIMessage，无message-start/delta/finish
        return [
            {
                "method": "messages",
                "params": {"data": [AIMessage(content="你好呀，好久没聊了！", id="msg-greet")]},
            }
        ]

    def _meta_events(self, events):
        return [e for e in events if e.event == "meta"]

    def test_whole_message_with_trace_emits_meta(self, monkeypatch):
        # 个性化问候有LLM调用→有trace，应下发meta
        monkeypatch.setattr(
            chat_service_module,
            "create_langfuse_handler",
            lambda: SimpleNamespace(last_trace_id="trace-greet"),
        )
        monkeypatch.setattr(chat_service_module, "maybe_schedule_eval", lambda **kw: None)

        service, _ = make_service(self.whole_message_events())
        request = ChatRequest(thread_id=uuid4(), message="你好")

        events = asyncio.run(consume(service, request))
        meta = self._meta_events(events)

        assert len(meta) == 1
        assert meta[0].data == {"message_id": "msg-greet"}

    def test_whole_message_without_trace_skips_meta(self, monkeypatch):
        # 固定寒暄无LLM调用→无trace，不下发meta（前端不渲染反馈按钮）
        monkeypatch.setattr(chat_service_module, "create_langfuse_handler", lambda: None)
        monkeypatch.setattr(chat_service_module, "maybe_schedule_eval", lambda **kw: None)

        service, _ = make_service(self.whole_message_events())
        request = ChatRequest(thread_id=uuid4(), message="谢谢")

        events = asyncio.run(consume(service, request))
        assert self._meta_events(events) == []

    def test_whole_message_still_yields_content(self, monkeypatch):
        # 完整消息仍正常推给前端
        monkeypatch.setattr(
            chat_service_module,
            "create_langfuse_handler",
            lambda: SimpleNamespace(last_trace_id="trace-greet"),
        )
        monkeypatch.setattr(chat_service_module, "maybe_schedule_eval", lambda **kw: None)

        service, _ = make_service(self.whole_message_events())
        request = ChatRequest(thread_id=uuid4(), message="你好")

        events = asyncio.run(consume(service, request))
        message_events = [e for e in events if e.event == "message"]
        assert message_events[0].data == "你好呀，好久没聊了！"

    def test_fixed_workflow_skips_meta_even_with_trace(self, monkeypatch):
        # human_handoff/fallback等固定话术即使有trace（route识别意图的LLM调用）也不开放反馈
        monkeypatch.setattr(
            chat_service_module,
            "create_langfuse_handler",
            lambda: SimpleNamespace(last_trace_id="trace-handoff"),
        )
        monkeypatch.setattr(chat_service_module, "maybe_schedule_eval", lambda **kw: None)

        for workflow in ("human_handoff", "fallback"):
            service, _ = make_service(
                self.whole_message_events(), active_workflow=workflow
            )
            request = ChatRequest(thread_id=uuid4(), message="我要投诉")

            events = asyncio.run(consume(service, request))
            assert self._meta_events(events) == []
