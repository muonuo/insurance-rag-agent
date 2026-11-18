"""app.core.observability Langfuse封装测试：开关行为、每轮独立handler与配置片段"""

from app.core.config import settings
from app.core.observability import (
    create_langfuse_handler,
    flush_langfuse,
    langfuse_enabled,
    observe_config,
    warm_langfuse,
)


def _enable(monkeypatch, enabled: bool = True, public: str = "pk-test", secret: str = "sk-test"):
    monkeypatch.setattr(settings.observability, "enabled", enabled)
    monkeypatch.setattr(settings.observability, "public_key", public)
    monkeypatch.setattr(settings.observability, "secret_key", secret)


class TestLangfuseDisabled:
    def test_disabled_returns_none_and_empty_config(self, monkeypatch):
        _enable(monkeypatch, enabled=False)
        assert langfuse_enabled() is False
        assert create_langfuse_handler() is None
        # 空dict展开进config不影响任何行为
        assert observe_config(1, "t-1") == {}

    def test_enabled_but_missing_key_is_still_none(self, monkeypatch):
        # 只开开关没配key视为未配置，不能半启用
        _enable(monkeypatch, enabled=True, public="", secret="")
        assert langfuse_enabled() is False
        assert create_langfuse_handler() is None

    def test_flush_disabled_is_noop(self, monkeypatch):
        _enable(monkeypatch, enabled=False)
        flush_langfuse()

    def test_warm_disabled_is_noop(self, monkeypatch):
        _enable(monkeypatch, enabled=False)
        warm_langfuse()


class TestLangfuseEnabled:
    def test_handlers_created_per_call_are_distinct(self, monkeypatch):
        _enable(monkeypatch)

        first = create_langfuse_handler()
        second = create_langfuse_handler()

        assert first is not None
        assert second is not None
        # 每轮独立handler：并发请求的last_trace_id互不覆盖
        assert first is not second

    def test_last_trace_id_isolated_between_handlers(self, monkeypatch):
        _enable(monkeypatch)

        first = create_langfuse_handler()
        second = create_langfuse_handler()
        first.last_trace_id = "trace-a"

        assert second.last_trace_id is None

    def test_observe_config_with_explicit_handler(self, monkeypatch):
        _enable(monkeypatch)
        handler = create_langfuse_handler()

        fragment = observe_config(7, "thread-abc", handler=handler)

        assert fragment["run_name"] == "chat-turn"
        assert fragment["callbacks"] == [handler]
        # SDK v3+ 标准metadata键：看板按用户/会话聚合
        assert fragment["metadata"] == {
            "langfuse_user_id": "7",
            "langfuse_session_id": "thread-abc",
            "langfuse_tags": ["insurance-agent"],
        }

    def test_observe_config_creates_handler_when_omitted(self, monkeypatch):
        _enable(monkeypatch)

        fragment = observe_config(7, "thread-abc")

        assert len(fragment["callbacks"]) == 1

    def test_handlers_share_nothing_but_config_works(self, monkeypatch):
        # 两次独立调用生成的config不共享handler，验证并发路径不会串
        _enable(monkeypatch)
        first = observe_config(1, "thread-1")
        second = observe_config(2, "thread-2")

        assert first["callbacks"][0] is not second["callbacks"][0]
        assert first["metadata"]["langfuse_session_id"] == "thread-1"
        assert second["metadata"]["langfuse_session_id"] == "thread-2"
