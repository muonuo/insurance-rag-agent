"""答题卡：调用生产 Agent 逐题作答，同时捕获检索上下文 → ragas.Dataset。

每道题都会真实走一遍「意图识别 → 保险顾问子Agent → 条款检索工具」，
`retrieved_contexts` 来自 app.rag.capture 的捕获钩子，即工具实际喂给
大模型、并展示给用户的那些条款片段。
"""

import json
import time

import rag_eval.compat  # noqa: F401  必须先于 ragas 加载

from langchain_core.messages import AIMessage, HumanMessage
from ragas import Dataset

# 生产链路采用函数内惰性导入：导入了 app.agents 就会初始化 Milvus 连接，
# 放在模块顶层会让题库校验这类离线操作也依赖基础设施可用。
from rag_eval.config import (
    DATASET_NAME,
    EVAL_USER_ID,
    QUESTIONS_PATH,
    REQUEST_DELAY,
    RESULTS_DIR,
)

_DATASET_CSV = RESULTS_DIR / "datasets" / f"{DATASET_NAME}.csv"


def load_questions(limit: int | None = None) -> list[dict]:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    _validate_questions(questions)
    return questions[:limit] if limit else questions


def _validate_questions(questions: list[dict]) -> None:
    """标准答案是 Context Recall 的评分尺子，缺失就没法判卷"""
    if not questions:
        raise SystemExit(f"题库为空: {QUESTIONS_PATH}")

    missing = [
        str(i + 1)
        for i, item in enumerate(questions)
        if not str(item.get("ground_truth", "")).strip()
    ]
    if missing:
        raise SystemExit(
            f"以下题目的 ground_truth 为空（行号 {', '.join(missing)}）：{QUESTIONS_PATH}\n"
            "请先从条款原文提炼标准答案再运行评测。"
        )


def load_cached_dataset() -> Dataset | None:
    """有缓存答题卡直接复用（跑判卷不用重新答题）"""
    if not _DATASET_CSV.exists():
        return None
    dataset = _open_dataset()
    dataset.reload()
    return dataset if len(dataset) > 0 else None


def _open_dataset() -> Dataset:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return Dataset(
        name=DATASET_NAME,
        backend="local/csv",
        root_dir=str(RESULTS_DIR),
    )


def _last_answer_text(result: dict) -> str:
    """取最终可展示的AI回复（跳过工具调用消息）"""
    for message in reversed(result.get("messages", [])):
        if not isinstance(message, AIMessage):
            continue
        if message.tool_calls:
            continue
        if message.text:
            return message.text
    return ""


async def build_dataset(
    limit: int | None = None,
    rebuild: bool = False,
) -> Dataset:
    """生成答题卡；已有缓存则复用，rebuild=True 强制重答"""

    if rebuild:
        _DATASET_CSV.unlink(missing_ok=True)

    dataset = _open_dataset()
    if _DATASET_CSV.exists():
        dataset.reload()
    if len(dataset) > 0:
        print(
            f"加载了 {len(dataset)} 条缓存答题卡，跳过重新答题"
            "（想重答加 --rebuild）"
        )
        return dataset

    questions = load_questions(limit)
    print(f"开始答题并采集检索上下文（共 {len(questions)} 题，间隔 {REQUEST_DELAY}s）...")
    print("-" * 60)

    # 校验通过后再初始化生产链路
    from app.agents.orchestrator import init_insurance_agent
    from app.agents.schemas import InsuranceAgentContext
    from app.core.observability import create_langfuse_handler, flush_langfuse, langfuse_enabled
    from app.infra.checkpointer import close_checkpointer, init_checkpointer
    from app.infra.database import close_database
    from app.infra.store import init_store
    from app.rag.capture import capture_contexts

    checkpointer = await init_checkpointer()
    # Store与Checkpointer共用连接池，必须在其之后初始化
    store = await init_store()
    agent = await init_insurance_agent(checkpointer, store)
    thread_ids: list[str] = []
    tracing_enabled = langfuse_enabled()
    if tracing_enabled:
        print("Langfuse追踪已启用：每题将生成独立trace，判卷分数会回写到对应trace")

    try:
        for i, item in enumerate(questions):
            question = item["question"]
            thread_id = f"rag-eval-{i}-{int(time.time() * 1000)}"
            thread_ids.append(thread_id)

            # 每题独立handler：last_trace_id只服务本题，避免题间串线
            handler = create_langfuse_handler()
            config: dict = {"configurable": {"thread_id": thread_id}}
            if handler is not None:
                config.update(
                    {
                        "callbacks": [handler],
                        "run_name": "rag-eval-question",
                        "metadata": {
                            "langfuse_user_id": str(EVAL_USER_ID),
                            "langfuse_session_id": thread_id,
                            "langfuse_tags": ["rag-eval", "offline-eval"],
                        },
                    }
                )

            # 开启上下文捕获 → 调用生产Agent → 拿到工具真实使用的条款片段
            with capture_contexts() as bucket:
                result = await agent.ainvoke(
                    {"messages": [HumanMessage(content=question)]},
                    config,
                    context=InsuranceAgentContext(
                        user_id=EVAL_USER_ID, thread_id=thread_id
                    ),
                )

            contexts = list(dict.fromkeys(bucket))  # 去重且保持检索顺序
            response = _last_answer_text(result)
            trace_id = getattr(handler, "last_trace_id", None) if handler is not None else None
            if not response:
                print(f"  ⚠️ 第 {i + 1} 题未取得AI回复（可能触发了中断），已按空回答记录")

            dataset.append(
                {
                    "user_input": question,
                    "response": response,
                    "retrieved_contexts": contexts,
                    "reference": item["ground_truth"],
                    # 判卷后据此回写分数；无追踪时为空串
                    "trace_id": trace_id or "",
                }
            )
            print(
                f"[{i + 1}/{len(questions)}] {question[:40]}... "
                f"OK (contexts: {len(contexts)}条, answer: {len(response)}字, "
                f"trace: {'有' if trace_id else '无'})"
            )

            if i < len(questions) - 1:
                time.sleep(REQUEST_DELAY)
    finally:
        # 清理评测产生的临时会话，避免污染 checkpointer
        for thread_id in thread_ids:
            try:
                await agent.checkpointer.adelete_thread(thread_id)
            except Exception:
                pass
        # 把trace和后续分数刷进Langfuse（未启用时空操作）
        flush_langfuse()
        await close_checkpointer()
        await close_database()

    dataset.save()
    print(f"\n答题卡已保存: {len(questions)} 条 -> {_DATASET_CSV}")
    return dataset
