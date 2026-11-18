"""Langfuse可观测性封装：业务代码只面对observe_config，未启用时零开销

启用条件（.env）：LANGFUSE_ENABLED=true 且 public/secret key 齐全。
追踪原理：CallbackHandler随config.callbacks传播进子Agent与工具调用，
用config.metadata里的langfuse_*键给trace打用户/会话标签（SDK v3+标准用法）。

注意：CallbackHandler按轮次创建（不是全局单例）。handler上的last_trace_id
会被后续run覆盖，全局单例在并发对话下会把A轮的trace_id串到B轮；
底层client与上报队列仍是进程级单例，按轮创建handler没有额外开销。
"""

import os
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_warmed = False


def langfuse_enabled() -> bool:
    """追踪是否具备启用条件（开关打开且双key齐全）"""

    obs = settings.observability
    return obs.enabled and bool(obs.public_key) and bool(obs.secret_key)


def _sync_langfuse_env() -> None:
    """SDK从环境变量读连接配置，把settings同步过去（pydantic只读.env不导出）"""

    obs = settings.observability
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", obs.public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", obs.secret_key)
    os.environ.setdefault("LANGFUSE_HOST", obs.host)


def create_langfuse_handler() -> Any | None:
    """为本轮对话创建独立handler；未启用或创建失败返回None"""

    if not langfuse_enabled():
        return None
    try:
        _sync_langfuse_env()
        # 惰性导入：应用启动不依赖langfuse包
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception:
        logger.exception("Langfuse handler创建失败，本轮不追踪")
        return None


def warm_langfuse() -> None:
    """启动预热：初始化全局client并打印一次启用日志；失败只降级不影响启动"""

    global _warmed
    if _warmed or not langfuse_enabled():
        return
    try:
        _sync_langfuse_env()
        from langfuse import get_client

        get_client()
        _warmed = True
        logger.info("Langfuse追踪已启用", host=settings.observability.host)
    except Exception:
        logger.exception("Langfuse初始化失败，追踪降级为关闭")


def observe_config(
    user_id: int,
    thread_id: str,
    handler: Any | None = None,
) -> dict:
    """返回要合并进LangGraph config的可观测性片段；未启用时为空dict

    用法：handler = create_langfuse_handler()
          config = {"configurable": {...}, **observe_config(user_id, thread_id, handler)}
    """

    if handler is None:
        handler = create_langfuse_handler()
    if handler is None:
        return {}
    return {
        "callbacks": [handler],
        "run_name": "chat-turn",
        "metadata": {
            "langfuse_user_id": str(user_id),
            "langfuse_session_id": thread_id,
            "langfuse_tags": ["insurance-agent"],
        },
    }


def get_langfuse_client() -> Any | None:
    """返回已同步凭证的Langfuse client；未启用或初始化失败返回None

    独立脚本（如离线判卷）不经过 warm_langfuse，必须走这里把settings
    里的key/host同步到环境变量后再取client，否则create_score会因无凭证失败。
    """

    if not langfuse_enabled():
        return None
    try:
        _sync_langfuse_env()
        from langfuse import get_client

        return get_client()
    except Exception:
        logger.exception("Langfuse client初始化失败")
        return None


def flush_langfuse() -> None:
    """关服前把批量队列里的trace冲出去；未启用或失败时不影响停机"""

    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        logger.exception("Langfuse flush失败")
