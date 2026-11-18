"""离线评测分数回写Langfuse：按答题卡行里的trace_id挂到对应trace

答题卡在build阶段为每题创建了独立trace（rag-eval-question），trace_id随行保存；
判卷完成后调用upload_scores即可在Langfuse看板按分数筛选离线评测结果。
trace_id缺失（旧缓存答题卡）时只告警不报错，提示用--rebuild重新答题。
"""

from typing import Any, Mapping

from app.core.logging import get_logger

logger = get_logger(__name__)

SCORE_COMMENT = "offline_eval"


def trace_id_from_row(row: Mapping[str, Any]) -> str | None:
    """从答题卡行中取trace_id；缺失/空值/pandas的NaN都视为没有"""

    try:
        value = row.get("trace_id")
    except AttributeError:
        return None
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def upload_scores(
    row: Mapping[str, Any],
    scores: Mapping[str, float],
    *,
    client: Any | None = None,
    comment: str = SCORE_COMMENT,
) -> int:
    """把一条答题记录的指标分数上传到Langfuse，返回成功条数

    任何失败都只记日志，不影响判卷流程。
    """

    if not scores:
        return 0

    trace_id = trace_id_from_row(row)
    if not trace_id:
        logger.warning(
            "离线评测分数未上传：答题卡缺少trace_id（请用 build --rebuild 重新答题）"
        )
        return 0

    if client is None:
        # 走封装：会把settings里的Langfuse key/host同步到环境变量（独立脚本必需）
        from app.core.observability import get_langfuse_client

        client = get_langfuse_client()
        if client is None:
            logger.warning("Langfuse未启用或初始化失败，离线评测分数未上传")
            return 0

    uploaded = 0
    for name, value in scores.items():
        try:
            client.create_score(
                name=name,
                value=float(value),
                trace_id=trace_id,
                data_type="NUMERIC",
                comment=comment,
            )
            uploaded += 1
        except Exception:
            logger.warning("离线评测分数上传失败", metric=name, trace_id=trace_id)

    logger.info(
        "离线评测分数已上传",
        trace_id=trace_id,
        uploaded=uploaded,
        total=len(scores),
    )
    return uploaded
