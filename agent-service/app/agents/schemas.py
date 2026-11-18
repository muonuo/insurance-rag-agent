from dataclasses import dataclass
from typing import Annotated, Any, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

# 路由意图常量
Intent = Literal[
    "chitchat",
    "recommendation_plan",
    "claim",
    "human_handoff",
    "fallback",
]


class RouteResult(BaseModel):
    """意图识别结果"""

    intent: Intent = Field(description="本轮消息应该进入的工作流")
    reason: str = Field(description="判断该意图的理由")


@dataclass(frozen=True)
class InsuranceAgentContext:
    """保险顾问Agent运行时上下文，由后端传入，不暴露给大模型"""

    user_id: int
    # 会话id：会话短期记忆（Redis缓存+摘要压缩）按它隔离
    thread_id: str | None = None


AdditionalInfo = dict[str, Any]


def merge_additional_info(
    left: dict[str, AdditionalInfo],
    right: dict[str, AdditionalInfo],
) -> dict[str, AdditionalInfo]:
    """按照AI消息id合并引用信息（不可变合并）"""
    if not left:
        return right
    return {**left, **right}


def merge_trace_ids(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    """按照AI消息id合并消息→Langfuse trace映射（不可变合并）"""
    if not left:
        return right
    return {**left, **right}


class InsuranceAgentState(MessagesState):
    """保险顾问主Graph状态"""

    previous_workflow: Intent
    active_workflow: Intent
    # 记录AI回答中的引用来源信息，key为AI消息id
    additional_info: Annotated[
        dict[str, AdditionalInfo],
        merge_additional_info,
    ]
    # AI消息id → 本轮Langfuse trace_id，用户反馈接口据此把分数挂到对应trace
    trace_ids: Annotated[
        dict[str, str],
        merge_trace_ids,
    ]