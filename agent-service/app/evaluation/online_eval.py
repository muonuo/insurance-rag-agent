"""在线RAG评测：抽样给真实对话打分，并把分数回写到Langfuse trace

流程：对话轮结束 → 采样命中且各项守卫通过 → 后台任务用ragas计算
faithfulness / answer_relevancy → create_score挂到本轮trace上。

约束：
- 顶层不导入ragas（首次后台任务才导入，不拖慢启动）
- 评测失败绝不影响对话，全部异常在后台任务内消化
- 每个指标独立降级：embedding不可用时仍上传faithfulness
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.core.observability import langfuse_enabled

logger = get_logger(__name__)

SCORE_FAITHFULNESS = "faithfulness"
SCORE_ANSWER_RELEVANCY = "answer_relevancy"
SCORE_COMMENT = "online_eval"

# 判卷LLM与离线评测保持一致（DeepSeek OpenAI兼容接口）
JUDGE_LLM_BASE_URL = "https://api.deepseek.com"
MAX_PENDING_TASKS = 20          # 单进程待执行评测上限，防止积压
MAX_CONTEXT_CHARS = 20_000      # 送判官的上下文总字符上限
JUDGE_RETRY_COOLDOWN_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class _Judges:
    llm: Any | None
    embeddings: Any | None


_judges: _Judges | None = None
_judges_ready = False
_judges_failed_at: float | None = None
_judge_lock = threading.Lock()

# 强引用后台任务，避免asyncio任务被GC中断
_background_tasks: set[asyncio.Task[None]] = set()


# ---------- 采样与预处理 ----------

def should_sample(sample_rate: float, roll: float | None = None) -> bool:
    """按采样率判定本轮是否评测；roll可注入便于测试"""

    if sample_rate <= 0:
        return False
    if sample_rate >= 1:
        return True
    return (random.random() if roll is None else roll) < sample_rate


def _prepare_contexts(contexts: list[str]) -> list[str]:
    """去重、去空、限制总字符，避免判官上下文过长"""

    seen: set[str] = set()
    result: list[str] = []
    total = 0
    for raw in contexts:
        text = (raw or "").strip()
        if not text or text in seen:
            continue
        remaining = MAX_CONTEXT_CHARS - total
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining]
        result.append(text)
        seen.add(text)
        total += len(text)
    return result


def _valid_score(value: Any) -> float | None:
    """校验分数为[0,1]内的有限数；非法值返回None"""

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        return None
    return score


# ---------- 判官惰性构建 ----------

def _get_judges() -> _Judges:
    """构建并缓存判官；导入ragas与初始化客户端都较重，仅在后台线程执行一次

    全部失败后进入冷却期，避免每轮评测都重复导入并刷日志。
    """

    global _judges, _judges_ready, _judges_failed_at

    with _judge_lock:
        if _judges_ready and _judges is not None:
            return _judges
        if (
            _judges_failed_at is not None
            and time.monotonic() - _judges_failed_at < JUDGE_RETRY_COOLDOWN_SECONDS
        ):
            return _Judges(llm=None, embeddings=None)

        try:
            # 关闭ragas自带的匿名遥测（默认会向t.explodinggradients.com上报）
            os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
            import app.evaluation.ragas_compat  # noqa: F401  必须先于ragas
            from openai import AsyncOpenAI
            from ragas.embeddings.base import embedding_factory
            from ragas.llms import llm_factory
        except Exception:
            _judges_failed_at = time.monotonic()
            logger.exception("在线评测判官依赖加载失败")
            return _Judges(llm=None, embeddings=None)

        llm = None
        embeddings = None

        try:
            llm_client = AsyncOpenAI(
                api_key=settings.llm.api_key, base_url=JUDGE_LLM_BASE_URL
            )
            llm = llm_factory(
                model=settings.llm.chat_model,
                client=llm_client,
                temperature=0,
                max_tokens=8192,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception:
            logger.exception("在线评测LLM判官初始化失败")

        try:
            emb_client = AsyncOpenAI(
                api_key="ollama",
                base_url=settings.online_eval.embedding_base_url,
            )
            embeddings = embedding_factory(
                model=settings.online_eval.embedding_model, client=emb_client
            )
        except Exception:
            logger.exception("在线评测embedding判官初始化失败")

        if llm is None and embeddings is None:
            _judges_failed_at = time.monotonic()
            return _Judges(llm=None, embeddings=None)

        _judges = _Judges(llm=llm, embeddings=embeddings)
        _judges_ready = True
        return _judges


# ---------- 指标计算 ----------

async def _score_faithfulness(
    judges: _Judges, *, user_input: str, answer: str, contexts: list[str]
) -> float:
    from ragas.metrics.collections import Faithfulness

    result = await Faithfulness(llm=judges.llm).ascore(
        user_input=user_input,
        response=answer,
        retrieved_contexts=contexts,
    )
    return float(result.value)


async def _score_answer_relevancy(
    judges: _Judges, *, user_input: str, answer: str
) -> float:
    from ragas.metrics.collections import AnswerRelevancy

    result = await AnswerRelevancy(
        llm=judges.llm, embeddings=judges.embeddings
    ).ascore(user_input=user_input, response=answer)
    return float(result.value)


# ---------- 评测与上传 ----------

async def evaluate_turn(
    *,
    user_input: str,
    answer: str,
    contexts: list[str],
    trace_id: str,
    user_id: int,
    thread_id: str,
    client: Any | None = None,
) -> dict[str, float]:
    """计算并上传单个对话轮的评测分数；返回成功上传的指标"""

    started = time.perf_counter()
    prepared = _prepare_contexts(contexts)
    if not prepared:
        return {}

    # 导入ragas与构建判官放到线程里，避免阻塞事件循环
    judges = await asyncio.to_thread(_get_judges)
    scores: dict[str, float] = {}

    if judges.llm is not None:
        try:
            value = _valid_score(
                await _score_faithfulness(
                    judges,
                    user_input=user_input,
                    answer=answer,
                    contexts=prepared,
                )
            )
            if value is None:
                logger.warning("在线评测指标无效", metric=SCORE_FAITHFULNESS, trace_id=trace_id)
            else:
                scores[SCORE_FAITHFULNESS] = value
        except Exception:
            logger.warning(
                "在线评测指标失败", metric=SCORE_FAITHFULNESS, trace_id=trace_id
            )

    # embedding不可用时answer_relevancy失败，但faithfulness照常上传
    if judges.llm is not None and judges.embeddings is not None:
        try:
            value = _valid_score(
                await _score_answer_relevancy(
                    judges, user_input=user_input, answer=answer
                )
            )
            if value is None:
                logger.warning("在线评测指标无效", metric=SCORE_ANSWER_RELEVANCY, trace_id=trace_id)
            else:
                scores[SCORE_ANSWER_RELEVANCY] = value
        except Exception:
            logger.warning(
                "在线评测指标失败", metric=SCORE_ANSWER_RELEVANCY, trace_id=trace_id
            )

    if scores:
        try:
            if client is None:
                from langfuse import get_client

                client = get_client()
            for name, value in scores.items():
                try:
                    client.create_score(
                        name=name,
                        value=value,
                        trace_id=trace_id,
                        data_type="NUMERIC",
                        comment=SCORE_COMMENT,
                    )
                except Exception:
                    logger.warning("在线评测分数上传失败", metric=name, trace_id=trace_id)
        except Exception:
            logger.exception("Langfuse client不可用，分数未上传", trace_id=trace_id)

    logger.info(
        "在线评测完成",
        trace_id=trace_id,
        user_id=user_id,
        thread_id=thread_id,
        contexts=len(prepared),
        answer_chars=len(answer),
        scores={k: round(v, 4) for k, v in scores.items()},
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return scores


# ---------- 调度与守卫 ----------

def _log_skip(reason: str, trace_id: str | None, thread_id: str | None) -> None:
    logger.debug("在线评测跳过", reason=reason, trace_id=trace_id, thread_id=thread_id)


def _on_eval_done(task: asyncio.Task[None]) -> None:
    _background_tasks.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        logger.info("在线评测任务已取消")
    except Exception:
        logger.exception("在线评测任务结果异常")


async def _run_eval_safely(**kwargs: Any) -> None:
    try:
        await evaluate_turn(**kwargs)
    except Exception:
        logger.exception("在线评测任务异常", trace_id=kwargs.get("trace_id"))


def maybe_schedule_eval(
    *,
    user_input: str | None,
    answer: str | None,
    contexts: list[str] | None,
    trace_id: str | None,
    user_id: int,
    thread_id: str,
    has_interrupt: bool,
    is_decision: bool,
) -> asyncio.Task[None] | None:
    """守卫通过则把评测投递到后台任务；本函数绝不抛出异常"""

    try:
        cfg = settings.online_eval
        if not cfg.enabled:
            return None
        if is_decision:
            _log_skip("decision", trace_id, thread_id)
            return None
        if has_interrupt:
            _log_skip("interrupt", trace_id, thread_id)
            return None
        if not trace_id:
            _log_skip("trace_missing", trace_id, thread_id)
            return None
        if not langfuse_enabled():
            _log_skip("langfuse_disabled", trace_id, thread_id)
            return None
        if not contexts:
            _log_skip("no_contexts", trace_id, thread_id)
            return None
        if not answer or not answer.strip():
            _log_skip("empty_answer", trace_id, thread_id)
            return None
        if not user_input:
            _log_skip("empty_input", trace_id, thread_id)
            return None
        if not should_sample(cfg.sample_rate):
            _log_skip("not_sampled", trace_id, thread_id)
            return None
        if len(_background_tasks) >= MAX_PENDING_TASKS:
            _log_skip("too_many_pending", trace_id, thread_id)
            return None

        loop = asyncio.get_running_loop()
        task = loop.create_task(
            _run_eval_safely(
                user_input=user_input,
                answer=answer,
                contexts=contexts,
                trace_id=trace_id,
                user_id=user_id,
                thread_id=thread_id,
            ),
            name=f"online-eval-{trace_id}",
        )
        _background_tasks.add(task)
        task.add_done_callback(_on_eval_done)
        logger.info(
            "在线评测已调度",
            trace_id=trace_id,
            user_id=user_id,
            thread_id=thread_id,
            sample_rate=cfg.sample_rate,
            contexts=len(contexts),
            answer_chars=len(answer),
        )
        return task
    except Exception:
        # 调度异常绝不影响对话
        logger.exception("在线评测调度异常")
        return None


async def drain_online_evals(timeout: float = 5.0) -> None:
    """关停前等待在途评测，超时取消；必须早于flush_langfuse调用"""

    pending = [task for task in _background_tasks if not task.done()]
    if not pending:
        return
    _done, still_pending = await asyncio.wait(pending, timeout=timeout)
    for task in still_pending:
        task.cancel()
    if still_pending:
        logger.warning("在线评测关停超时", pending=len(still_pending))


def log_online_eval_config() -> None:
    """启动时打印评测配置状态"""

    cfg = settings.online_eval
    if not cfg.enabled:
        return
    if not langfuse_enabled():
        logger.warning("ONLINE_EVAL_ENABLED=true 但 Langfuse 未配置，评测将被跳过")
        return
    logger.info(
        "在线评测已启用",
        sample_rate=cfg.sample_rate,
        embedding_model=cfg.embedding_model,
    )
