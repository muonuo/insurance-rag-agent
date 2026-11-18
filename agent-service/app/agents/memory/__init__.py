"""Agent 记忆：会话短期记忆（Redis 缓存 + 摘要压缩）"""

from app.core.config import settings

from .middleware import SessionMemoryMiddleware
from .session_memory import (
    SessionMemory,
    SessionSnapshot,
    split_history_and_current,
)

# 全进程单例：中间件与对话服务共用同一份缓存与模型实例
session_memory = SessionMemory(
    url=settings.session_memory.redis_url,
    ttl=settings.session_memory.ttl,
    recent_limit=settings.session_memory.recent_messages,
    enabled=settings.session_memory.enabled,
)

__all__ = [
    "SessionMemory",
    "SessionMemoryMiddleware",
    "SessionSnapshot",
    "session_memory",
    "split_history_and_current",
]
