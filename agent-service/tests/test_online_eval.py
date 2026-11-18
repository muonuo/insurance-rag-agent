"""app.evaluation.online_eval 在线评测测试：采样、守卫、指标降级与分数上传"""

import asyncio

import app.evaluation.online_eval as online_eval
from app.core.config import settings
from app.evaluation.online_eval import (
    SCORE_ANSWER_RELEVANCY,
    SCORE_FAITHFULNESS,
    _Judges,
    _prepare_contexts,
    _valid_score,
    drain_online_evals,
    evaluate_turn,
    maybe_schedule_eval,
    should_sample,
)


class FakeClient:
    def __init__(self, fail: bool = False):
        self.scores: list[dict] = []
        self.fail = fail

    def create_score(self, **kwargs):
        if self.fail:
            raise RuntimeError("upload failed")
        self.scores.append(kwargs)


def _enable_eval(monkeypatch, rate: float = 1.0, langfuse: bool = True):
    monkeypatch.setattr(settings.online_eval, "enabled", True)
    monkeypatch.setattr(settings.online_eval, "sample_rate", rate)
    monkeypatch.setattr(settings.observability, "enabled", langfuse)
    monkeypatch.setattr(settings.observability, "public_key", "pk-test" if langfuse else "")
    monkeypatch.setattr(settings.observability, "secret_key", "sk-test" if langfuse else "")


def _fake_judges(monkeypatch, *, llm=True, embeddings=True):
    monkeypatch.setattr(
        online_eval,
        "_get_judges",
        lambda: _Judges(llm=object() if llm else None, embeddings=object() if embeddings else None),
    )


def _fake_metrics(monkeypatch, faithfulness=0.9, relevancy=0.8, raise_faithfulness=False, raise_relevancy=False):
    async def score_faithfulness(judges, **kwargs):
        if raise_faithfulness:
            raise RuntimeError("judge down")
        return faithfulness

    async def score_relevancy(judges, **kwargs):
        if raise_relevancy:
            raise RuntimeError("embedding down")
        return relevancy

    monkeypatch.setattr(online_eval, "_score_faithfulness", score_faithfulness)
    monkeypatch.setattr(online_eval, "_score_answer_relevancy", score_relevancy)


class TestSampling:
    def test_boundaries(self):
        assert should_sample(0, roll=0.5) is False
        assert should_sample(1, roll=0.99) is True
        assert should_sample(0.3, roll=0.29) is True
        assert should_sample(0.3, roll=0.3) is False


class TestHelpers:
    def test_contexts_dedup_and_truncate(self):
        long_text = "长" * (online_eval.MAX_CONTEXT_CHARS + 100)
        result = _prepare_contexts(["a", "a", "", "  ", long_text, "b"])
        assert result[0] == "a"
        assert result[1].startswith("长")
        # 前一条"a"已占1字符，剩余额度优先给长文本
        assert len(result[1]) == online_eval.MAX_CONTEXT_CHARS - len("a")
        # 已达上限，后续上下文被丢弃
        assert "b" not in result

    def test_valid_score(self):
        assert _valid_score(0.5) == 0.5
        assert _valid_score(0) == 0.0
        assert _valid_score(1) == 1.0
        assert _valid_score(float("nan")) is None
        assert _valid_score(1.5) is None
        assert _valid_score(-0.1) is None
        assert _valid_score("bad") is None


class TestGuards:
    def _call(self, **overrides):
        params = {
            "user_input": "问题",
            "answer": "回答",
            "contexts": ["条款"],
            "trace_id": "trace-1",
            "user_id": 1,
            "thread_id": "thread-1",
            "has_interrupt": False,
            "is_decision": False,
        }
        params.update(overrides)
        return maybe_schedule_eval(**params)

    def test_disabled(self, monkeypatch):
        monkeypatch.setattr(settings.online_eval, "enabled", False)
        assert self._call() is None

    def test_decision_turn(self, monkeypatch):
        _enable_eval(monkeypatch)
        assert self._call(is_decision=True) is None

    def test_interrupt_turn(self, monkeypatch):
        _enable_eval(monkeypatch)
        assert self._call(has_interrupt=True) is None

    def test_missing_trace_id(self, monkeypatch):
        _enable_eval(monkeypatch)
        assert self._call(trace_id=None) is None

    def test_langfuse_disabled(self, monkeypatch):
        _enable_eval(monkeypatch, langfuse=False)
        assert self._call() is None

    def test_no_contexts(self, monkeypatch):
        _enable_eval(monkeypatch)
        assert self._call(contexts=[]) is None

    def test_empty_answer(self, monkeypatch):
        _enable_eval(monkeypatch)
        assert self._call(answer="   ") is None

    def test_empty_input(self, monkeypatch):
        _enable_eval(monkeypatch)
        assert self._call(user_input=None) is None

    def test_not_sampled(self, monkeypatch):
        _enable_eval(monkeypatch, rate=0.0)
        assert self._call() is None

    def test_pending_cap(self, monkeypatch):
        _enable_eval(monkeypatch)
        # 守卫在获取事件循环之前返回，可同步验证
        online_eval._background_tasks.update(
            object() for _ in range(online_eval.MAX_PENDING_TASKS)  # type: ignore[arg-type]
        )
        try:
            assert self._call() is None
        finally:
            online_eval._background_tasks.clear()

    def test_schedule_creates_task(self, monkeypatch):
        _enable_eval(monkeypatch)
        executed: list[dict] = []

        async def fake_run(**kwargs):
            executed.append(kwargs)

        monkeypatch.setattr(online_eval, "_run_eval_safely", fake_run)

        async def scenario():
            task = self._call()
            assert task is not None
            await task
            return task

        asyncio.run(scenario())
        assert executed and executed[0]["trace_id"] == "trace-1"
        # 任务完成后从强引用集合释放
        assert online_eval._background_tasks == set()


class TestEvaluateTurn:
    def _run(self, monkeypatch, client=None, **metric_kwargs):
        _fake_judges(monkeypatch)
        _fake_metrics(monkeypatch, **metric_kwargs)
        client = client or FakeClient()
        scores = asyncio.run(
            evaluate_turn(
                user_input="免赔额多少",
                answer="免赔额1万元",
                contexts=["条款原文"],
                trace_id="trace-1",
                user_id=1,
                thread_id="thread-1",
                client=client,
            )
        )
        return scores, client

    def test_uploads_both_scores(self, monkeypatch):
        scores, client = self._run(monkeypatch)

        assert scores == {SCORE_FAITHFULNESS: 0.9, SCORE_ANSWER_RELEVANCY: 0.8}
        assert [s["name"] for s in client.scores] == [SCORE_FAITHFULNESS, SCORE_ANSWER_RELEVANCY]
        for score in client.scores:
            assert score["trace_id"] == "trace-1"
            assert score["data_type"] == "NUMERIC"

    def test_faithfulness_failure_still_uploads_relevancy(self, monkeypatch):
        scores, client = self._run(monkeypatch, raise_faithfulness=True)

        assert scores == {SCORE_ANSWER_RELEVANCY: 0.8}
        assert [s["name"] for s in client.scores] == [SCORE_ANSWER_RELEVANCY]

    def test_relevancy_failure_still_uploads_faithfulness(self, monkeypatch):
        scores, client = self._run(monkeypatch, raise_relevancy=True)

        assert scores == {SCORE_FAITHFULNESS: 0.9}
        assert [s["name"] for s in client.scores] == [SCORE_FAITHFULNESS]

    def test_invalid_scores_skipped(self, monkeypatch):
        scores, client = self._run(
            monkeypatch, faithfulness=float("nan"), relevancy=1.7
        )

        assert scores == {}
        assert client.scores == []

    def test_embedding_missing_skips_relevancy(self, monkeypatch):
        _fake_judges(monkeypatch, embeddings=False)
        _fake_metrics(monkeypatch)
        client = FakeClient()

        scores = asyncio.run(
            evaluate_turn(
                user_input="问题",
                answer="回答",
                contexts=["条款"],
                trace_id="trace-1",
                user_id=1,
                thread_id="thread-1",
                client=client,
            )
        )
        assert scores == {SCORE_FAITHFULNESS: 0.9}

    def test_upload_failure_does_not_raise(self, monkeypatch):
        scores, client = self._run(monkeypatch, client=FakeClient(fail=True))
        # 指标算出来了，但上传失败不影响返回和调用方
        assert scores == {SCORE_FAITHFULNESS: 0.9, SCORE_ANSWER_RELEVANCY: 0.8}
        assert client.scores == []

    def test_no_contexts_returns_early(self, monkeypatch):
        _fake_judges(monkeypatch)
        _fake_metrics(monkeypatch)
        scores = asyncio.run(
            evaluate_turn(
                user_input="问题",
                answer="回答",
                contexts=[],
                trace_id="trace-1",
                user_id=1,
                thread_id="thread-1",
                client=FakeClient(),
            )
        )
        assert scores == {}

    def test_judges_all_failed_returns_empty(self, monkeypatch):
        monkeypatch.setattr(online_eval, "_get_judges", lambda: _Judges(llm=None, embeddings=None))
        client = FakeClient()
        scores = asyncio.run(
            evaluate_turn(
                user_input="问题",
                answer="回答",
                contexts=["条款"],
                trace_id="trace-1",
                user_id=1,
                thread_id="thread-1",
                client=client,
            )
        )
        assert scores == {}
        assert client.scores == []


class TestDrain:
    def test_drain_cancels_after_timeout(self):
        async def scenario():
            task = asyncio.get_running_loop().create_task(asyncio.sleep(10))
            online_eval._background_tasks.add(task)
            await drain_online_evals(timeout=0.01)
            # cancel后需让循环跑一拍，任务状态才会落为cancelled
            await asyncio.sleep(0)
            assert task.cancelled()
            online_eval._background_tasks.clear()

        asyncio.run(scenario())
