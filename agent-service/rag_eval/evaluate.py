"""判卷：LLM 走 DeepSeek，向量走本地 Ollama，RAGAS 六指标 + @experiment 批量。

deepseek-flash 是混合思考模型；判卷统一关闭思考并放大 max_tokens，
避免思考内容污染判卷 JSON 导致 NaN。
"""

import ast
import os

# 关闭ragas匿名遥测（默认会向t.explodinggradients.com上报并拖慢判卷）
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

import rag_eval.compat  # noqa: F401  必须先于 ragas 加载

from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas import Dataset, experiment
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    NoiseSensitivity,
)

from rag_eval.config import (
    DATASET_NAME,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    JUDGE_EMBEDDING_BASE_URL,
    JUDGE_EMBEDDING_MODEL,
    JUDGE_LLM,
    RESULTS_DIR,
)
from rag_eval.upload import upload_scores
from app.core.observability import flush_langfuse

# build_judges() 赋值，@experiment 函数运行时引用
_judge_llm = None
_judge_embeddings = None


def _normalize_contexts(value) -> list[str]:
    """CSV 缓存反序列化坑：retrieved_contexts 取出来可能是 list 的 repr 字符串。

    RAGAS 指标要求 list[str]；传字符串进去会把每个字符当一条文档逐个判分。
    """
    if isinstance(value, list):
        return [str(c) for c in value]
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(c) for c in parsed]
    except (ValueError, SyntaxError):
        pass
    return [str(value)]


def build_judges() -> None:
    """LLM 评"读懂内容"的指标；embedding 评"算相似度"的指标"""
    global _judge_llm, _judge_embeddings
    llm_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    emb_client = AsyncOpenAI(api_key="ollama", base_url=JUDGE_EMBEDDING_BASE_URL)

    _judge_llm = llm_factory(
        model=JUDGE_LLM,
        client=llm_client,
        temperature=0,
        max_tokens=8192,
        extra_body={"thinking": {"type": "disabled"}},
    )
    _judge_embeddings = embedding_factory(
        model=JUDGE_EMBEDDING_MODEL, client=emb_client
    )


class FastResult(BaseModel):
    """fast 档：检索段 + 生成段各一条核心指标"""

    context_recall: float
    faithfulness: float


class FullResult(FastResult):
    """full 档：全 6 指标。noise_sensitivity 越低越好，其余越高越好"""

    context_precision: float
    context_entity_recall: float
    answer_relevancy: float
    noise_sensitivity: float


@experiment(experiment_model=FastResult)
async def run_fast(row) -> FastResult:
    user_input = row.get("user_input")
    reference = row.get("reference")
    retrieved_contexts = _normalize_contexts(row.get("retrieved_contexts"))
    response = row.get("response")

    cr = await ContextRecall(llm=_judge_llm).ascore(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    faith = await Faithfulness(llm=_judge_llm).ascore(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
    )
    scores = {"context_recall": cr.value, "faithfulness": faith.value}
    # 回写到本题trace（答题卡没有trace_id时只告警）
    upload_scores(row, scores)
    return FastResult(**scores)


@experiment(experiment_model=FullResult)
async def run_full(row) -> FullResult:
    user_input = row.get("user_input")
    reference = row.get("reference")
    retrieved_contexts = _normalize_contexts(row.get("retrieved_contexts"))
    response = row.get("response")

    cp = await ContextPrecision(llm=_judge_llm).ascore(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    cr = await ContextRecall(llm=_judge_llm).ascore(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    cer = await ContextEntityRecall(llm=_judge_llm).ascore(
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    faith = await Faithfulness(llm=_judge_llm).ascore(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
    )
    ar = await AnswerRelevancy(llm=_judge_llm, embeddings=_judge_embeddings).ascore(
        user_input=user_input,
        response=response,
    )
    ns = await NoiseSensitivity(llm=_judge_llm).ascore(
        user_input=user_input,
        response=response,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    scores = {
        "context_recall": cr.value,
        "faithfulness": faith.value,
        "context_precision": cp.value,
        "context_entity_recall": cer.value,
        "answer_relevancy": ar.value,
        "noise_sensitivity": ns.value,
    }
    # 回写到本题trace（答题卡没有trace_id时只告警）
    upload_scores(row, scores)
    return FullResult(**scores)


async def run_experiment(dataset, test_size: float, metrics: str, name: str) -> None:
    """抽样判卷并落盘成绩单。测试子集单独命名，不覆盖全量答题卡"""
    _train, test = dataset.train_test_split(test_size=test_size)
    print(f"抽样判卷: {len(test)}/{len(dataset)} 题（test_size={test_size}）")

    test_dataset = Dataset.from_pandas(
        test.to_pandas(),
        name=f"{DATASET_NAME}_test",
        backend="local/csv",
        root_dir=str(RESULTS_DIR),
    )
    runner = run_fast if metrics == "fast" else run_full
    await runner.arun(test_dataset, name=name)
    # 把判卷阶段产生的分数冲刷到Langfuse（未启用时空操作）
    flush_langfuse()
    print("=" * 60)
    print(f"成绩单已保存至: {RESULTS_DIR}/experiments/{name}.csv")


async def score_single(dataset, index: int = 0) -> None:
    """单题跑全 6 指标——批量前先冒烟，确认裁判都能出分"""
    row = dataset.__getitem__(index)
    user_input = row.get("user_input")
    reference = row.get("reference")
    retrieved_contexts = _normalize_contexts(row.get("retrieved_contexts"))
    response = row.get("response")

    print(f"========开始评估：input: {user_input}===========")

    cp = await ContextPrecision(llm=_judge_llm).ascore(
        user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts
    )
    print(f"context_precision:      {cp.value}（1/6）")
    cr = await ContextRecall(llm=_judge_llm).ascore(
        user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts
    )
    print(f"context_recall:         {cr.value}（2/6）")
    cer = await ContextEntityRecall(llm=_judge_llm).ascore(
        reference=reference, retrieved_contexts=retrieved_contexts
    )
    print(f"context_entity_recall:  {cer.value}（3/6）")
    faith = await Faithfulness(llm=_judge_llm).ascore(
        user_input=user_input, response=response, retrieved_contexts=retrieved_contexts
    )
    print(f"faithfulness:           {faith.value}（4/6）")
    ar = await AnswerRelevancy(llm=_judge_llm, embeddings=_judge_embeddings).ascore(
        user_input=user_input, response=response
    )
    print(f"answer_relevancy:       {ar.value}（5/6）")
    ns = await NoiseSensitivity(llm=_judge_llm).ascore(
        user_input=user_input,
        response=response,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    print(f"noise_sensitivity:      {ns.value}（6/6，越低越好）")
    print("========================评估完成==========================")
