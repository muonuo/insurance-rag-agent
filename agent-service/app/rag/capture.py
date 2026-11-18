"""RAG评测用的检索上下文捕获钩子。

生产环境不开启捕获时，`record_contexts()` 是空操作，没有额外开销。
评测脚本在调用 Agent 之前用 `capture_contexts()` 开启捕获，就能拿到
工具实际喂给大模型的所有条款片段，用于计算 RAGAS 的检索类指标。
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_capture_bucket: ContextVar[list[str] | None] = ContextVar(
    "rag_eval_contexts", default=None
)


def record_contexts(contents: list[str]) -> None:
    """记录一次检索命中的条款正文（仅在捕获开启时生效）"""

    bucket = _capture_bucket.get()
    if bucket is not None:
        bucket.extend(contents)


@contextmanager
def capture_contexts() -> Iterator[list[str]]:
    """开启上下文捕获，返回按命中顺序累积的列表；退出后自动关闭"""

    bucket: list[str] = []
    token = _capture_bucket.set(bucket)
    try:
        yield bucket
    finally:
        _capture_bucket.reset(token)
