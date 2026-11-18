"""ChatService 用户反馈测试：trace定位、权限校验与Langfuse分数写入"""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.modules.chat.exceptions import (
    FeedbackTargetNotFound,
    FeedbackUnavailable,
    MemoryUnavailable,
)
from app.modules.chat.schema import ChatFeedbackRequest
from app.modules.chat.service import ChatService
from app.modules.chat_thread.exceptions import ChatThreadNotFound


class FakeFeedbackClient:
    def __init__(self, fail: bool = False):
        self.scores: list[dict] = []
        self.flushed = 0
        self.fail = fail

    def create_score(self, **kwargs):
        if self.fail:
            raise RuntimeError("upload down")
        self.scores.append(kwargs)

    def flush(self):
        self.flushed += 1


class FakeState:
    def __init__(self, values):
        self.values = values


class FakeAgent:
    def __init__(self, trace_ids, store=None):
        self.state = FakeState({"trace_ids": trace_ids})
        self.store = store

    async def aget_state(self, config):
        return self.state


class FakeRepository:
    def __init__(self, owner_ok=True):
        self.owner_ok = owner_ok

    async def required_owner(self, user_id, thread_id):
        return SimpleNamespace(id=thread_id) if self.owner_ok else None


def make_service(trace_ids=None, owner_ok=True, store=None):
    service = ChatService(session=None, agent=FakeAgent(trace_ids or {}, store=store))
    service.repository = FakeRepository(owner_ok)
    return service


def make_request(rating="up", message_id="m-1", comment=None):
    return ChatFeedbackRequest(
        thread_id=uuid4(), message_id=message_id, rating=rating, comment=comment
    )


class TestSubmitFeedback:
    def test_up_creates_boolean_score_one(self):
        client = FakeFeedbackClient()
        service = make_service({"m-1": "trace-a"})

        asyncio.run(service.submit_feedback(1, make_request(), client=client))

        assert len(client.scores) == 1
        score = client.scores[0]
        assert score["name"] == "user_feedback"
        assert score["value"] == 1.0
        assert score["trace_id"] == "trace-a"
        assert score["data_type"] == "BOOLEAN"
        # 立即冲刷，看板即时可见
        assert client.flushed == 1

    def test_down_creates_boolean_score_zero(self):
        client = FakeFeedbackClient()
        service = make_service({"m-1": "trace-a"})

        asyncio.run(service.submit_feedback(1, make_request(rating="down"), client=client))

        assert client.scores[0]["value"] == 0.0

    def test_comment_forwarded(self):
        client = FakeFeedbackClient()
        service = make_service({"m-1": "trace-a"})

        asyncio.run(
            service.submit_feedback(
                1, make_request(comment="回答里免赔额说错了"), client=client
            )
        )

        assert client.scores[0]["comment"] == "回答里免赔额说错了"

    def test_not_owner_raises_thread_not_found(self):
        service = make_service({"m-1": "trace-a"}, owner_ok=False)

        with pytest.raises(ChatThreadNotFound):
            asyncio.run(service.submit_feedback(1, make_request(), client=FakeFeedbackClient()))

    def test_unknown_message_raises_target_not_found(self):
        # 功能上线前的旧消息没有trace映射
        service = make_service({})

        with pytest.raises(FeedbackTargetNotFound):
            asyncio.run(service.submit_feedback(1, make_request(), client=FakeFeedbackClient()))

    def test_upload_failure_raises_unavailable(self):
        service = make_service({"m-1": "trace-a"})

        with pytest.raises(FeedbackUnavailable):
            asyncio.run(
                service.submit_feedback(1, make_request(), client=FakeFeedbackClient(fail=True))
            )


class FakeProfileStore:
    def __init__(self, existing: dict):
        self.data = dict(existing)

    async def aget(self, namespace, key):
        value = self.data.get((namespace, key))
        return None if value is None else SimpleNamespace(value=value)

    async def adelete(self, namespace, key):
        self.data.pop((namespace, key), None)


class TestClearProfile:
    def test_deletes_profile_without_touching_threads(self):
        from app.agents.profile import PROFILE_KEY, profile_namespace

        namespace = profile_namespace(1)
        store = FakeProfileStore({(namespace, PROFILE_KEY): {"age": 30}})
        service = make_service(store=store)

        deleted = asyncio.run(service.clear_profile(1))

        assert deleted is True
        assert store.data == {}

    def test_missing_profile_is_idempotent(self):
        service = make_service(store=FakeProfileStore({}))
        # 没画像也返回False而不是报错，接口保持幂等
        assert asyncio.run(service.clear_profile(1)) is False

    def test_store_unavailable_raises(self):
        service = make_service(store=None)
        with pytest.raises(MemoryUnavailable):
            asyncio.run(service.clear_profile(1))
