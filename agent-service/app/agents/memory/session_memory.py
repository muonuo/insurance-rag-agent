"""会话短期记忆：Redis 缓存「最近消息 + 更早历史摘要」

定位：Redis 只是「发给模型的紧凑视图」的缓存，Checkpointer 仍是全量真源。
- 最近 N 条原文 + 更早内容的滚动摘要，TTL 自动过期
- 摘要失败/Redis 故障时 fail-open，退回窗口截断或全量历史，不影响对话
- 摘要按增量更新（只总结尚未覆盖的部分），并用结构化字段约束，减少信息丢失
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 攒够多少条"新溢出消息"才触发一次增量摘要，避免每轮都调模型
REFRESH_CHUNK = 2
# 摘要最大长度（字符），防止摘要本身膨胀
MAX_SUMMARY_CHARS = 800

SUMMARY_PROMPT = """
你是保险客服对话的摘要助手。请把较早的对话压缩成结构化摘要，供后续对话继续使用。

要求：
1. 只记录原文出现过的信息，不推测、不补充。
2. 必须保留：用户诉求、已确认事实（年龄/家庭/预算/产品名/保单号）、已做结论、未决问题。
3. 涉及数字、产品名称、条款编号时，使用原文措辞，不要改写。
4. 不要提及"摘要""对话"等元信息，直接输出内容。

输出格式（没有内容则写"无"）：
用户诉求：
已确认事实：
已做结论：
未决问题：
""".strip()


@dataclass(frozen=True)
class SessionSnapshot:
    """Redis 中的会话紧凑视图"""

    summary: str | None
    recent: list[dict[str, Any]]
    summarized_count: int


def _is_visible(message: BaseMessage) -> bool:
    """只保留可展示的对话消息（跳过工具调用与工具结果）"""

    if not isinstance(message, (HumanMessage, AIMessage)):
        return False
    if getattr(message, "tool_calls", None):
        return False
    return bool(message.text)


def _message_to_dict(message: BaseMessage) -> dict[str, Any]:
    return {
        "role": "human" if isinstance(message, HumanMessage) else "ai",
        "content": message.text,
        "id": message.id,
    }


def _dict_to_message(data: dict[str, Any]) -> BaseMessage:
    cls = HumanMessage if data.get("role") == "human" else AIMessage
    return cls(content=data.get("content", ""), id=data.get("id"))


def split_history_and_current(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """按最后一条用户消息切开：之前是历史，之后（含本轮工具调用）是当前轮"""

    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[:index], messages[index:]
    return [], messages


class SessionMemory:
    """会话短期记忆缓存；Redis 不可用时全部降级为 None（调用方走兜底）"""

    def __init__(self, url: str, ttl: int, recent_limit: int, enabled: bool = True):
        self._url = url
        self._ttl = ttl
        self.recent_limit = recent_limit
        self.enabled = enabled
        self._client = None
        self._model = None
        # 同一会话的刷新串行化，避免并发触发多次摘要
        self._locks: dict[str, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    # ---------- 基础设施 ----------

    def _get_client(self):
        if self._client is None:
            import redis.asyncio as redis

            self._client = redis.Redis.from_url(
                self._url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
        return self._client

    def _keys(self, thread_id: str) -> tuple[str, str, str]:
        return (
            f"session:{thread_id}:summary",
            f"session:{thread_id}:recent",
            f"session:{thread_id}:meta",
        )

    def _get_model(self):
        if self._model is None:
            from langchain.chat_models import init_chat_model

            self._model = init_chat_model(
                settings.llm.chat_model,
                api_key=settings.llm.api_key,
                extra_body={"thinking": {"type": "disabled"}},
                max_tokens=800,
            )
        return self._model

    # ---------- 读写 ----------

    async def get(self, thread_id: str) -> SessionSnapshot | None:
        """读缓存；未命中/禁用/故障返回None"""

        if not self.enabled:
            return None
        try:
            summary_key, recent_key, meta_key = self._keys(thread_id)
            client = self._get_client()
            raw_summary, raw_recent, raw_meta = await client.mget(
                summary_key, recent_key, meta_key
            )
            if raw_recent is None:
                return None
            return SessionSnapshot(
                summary=raw_summary or None,
                recent=json.loads(raw_recent),
                summarized_count=int(json.loads(raw_meta or "{}").get("summarized_count", 0)),
            )
        except Exception as exc:
            logger.warning("会话记忆读取失败", thread_id=thread_id, error=str(exc)[:120])
            return None

    async def set(
        self,
        thread_id: str,
        summary: str | None,
        recent: list[dict[str, Any]],
        summarized_count: int,
    ) -> None:
        if not self.enabled:
            return
        try:
            summary_key, recent_key, meta_key = self._keys(thread_id)
            client = self._get_client()
            await client.setex(summary_key, self._ttl, summary or "")
            await client.setex(
                recent_key, self._ttl, json.dumps(recent, ensure_ascii=False)
            )
            await client.setex(
                meta_key,
                self._ttl,
                json.dumps({"summarized_count": summarized_count}),
            )
        except Exception as exc:
            logger.warning("会话记忆写入失败", thread_id=thread_id, error=str(exc)[:120])

    async def append_turn(self, thread_id: str, user_text: str, answer_text: str) -> None:
        """一轮结束后把问答追加进最近窗口（不触发摘要，摘要由中间件按需刷新）"""

        if not self.enabled or not user_text.strip() or not answer_text.strip():
            return
        try:
            snapshot = await self.get(thread_id)
            recent = list(snapshot.recent) if snapshot else []
            recent.append({"role": "human", "content": user_text, "id": None})
            recent.append({"role": "ai", "content": answer_text[:2000], "id": None})
            recent = recent[-self.recent_limit :]
            await self.set(
                thread_id,
                snapshot.summary if snapshot else None,
                recent,
                snapshot.summarized_count if snapshot else 0,
            )
        except Exception as exc:
            logger.warning("会话记忆追加失败", thread_id=thread_id, error=str(exc)[:120])

    async def invalidate(self, thread_id: str) -> None:
        """删除会话（或用户清除）时清空缓存"""

        if not self.enabled:
            return
        try:
            summary_key, recent_key, meta_key = self._keys(thread_id)
            await self._get_client().delete(summary_key, recent_key, meta_key)
        except Exception as exc:
            logger.warning("会话记忆清理失败", thread_id=thread_id, error=str(exc)[:120])

    # ---------- 摘要 ----------

    async def summarize(
        self, previous: str | None, messages: list[BaseMessage]
    ) -> str | None:
        """把已有摘要与新增的旧消息合并成新摘要；失败返回None（调用方保留旧摘要）"""

        if not messages:
            return previous
        transcript = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else '助手'}：{m.text}"
            for m in messages
        )
        prompt = (
            f"已有摘要（可能为空）：\n{previous or '无'}\n\n"
            f"需要合并进摘要的新增对话：\n{transcript}\n\n"
            "请输出合并后的完整摘要。"
        )
        try:
            from langchain_core.messages import HumanMessage as _Human

            response = await self._get_model().ainvoke(prompt)
            summary = (response.text or "").strip()
            if not summary:
                return previous
            return summary[:MAX_SUMMARY_CHARS]
        except Exception:
            logger.exception("会话摘要生成失败，保留旧摘要")
            return previous

    async def refresh(self, thread_id: str, history: list[BaseMessage]) -> None:
        """根据全量历史重建缓存：最近N条原文 + 更早部分的增量摘要"""

        texts = [m for m in history if _is_visible(m)]
        if not texts:
            return
        recent = texts[-self.recent_limit :]
        older = texts[: -self.recent_limit] if len(texts) > self.recent_limit else []

        snapshot = await self.get(thread_id)
        covered = min(snapshot.summarized_count, len(older)) if snapshot else 0
        summary = snapshot.summary if snapshot else None
        new_part = older[covered:]
        if new_part:
            try:
                summary = await self.summarize(summary, new_part)
            except Exception:
                # 摘要失败保留旧摘要，绝不能影响对话
                logger.exception("会话摘要失败，保留旧摘要", thread_id=thread_id)

        await self.set(
            thread_id,
            summary,
            [_message_to_dict(m) for m in recent],
            len(older),
        )
        logger.info(
            "会话记忆已刷新",
            thread_id=thread_id,
            recent=len(recent),
            older=len(older),
            summarized=len(new_part),
        )

    def maybe_schedule_refresh(
        self,
        thread_id: str,
        history: list[BaseMessage],
        snapshot: SessionSnapshot | None,
    ) -> None:
        """历史中出现新的溢出内容时，后台异步刷新（不阻塞本轮对话）"""

        if not self.enabled:
            return
        older_count = max(0, len([m for m in history if _is_visible(m)]) - self.recent_limit)
        if older_count <= 0:
            return
        if snapshot is not None and older_count - snapshot.summarized_count < REFRESH_CHUNK:
            return

        lock = self._locks.setdefault(thread_id, asyncio.Lock())
        if lock.locked():
            return

        async def run() -> None:
            async with lock:
                try:
                    await self.refresh(thread_id, history)
                except Exception:
                    logger.exception("会话记忆后台刷新失败", thread_id=thread_id)

        try:
            task = asyncio.get_running_loop().create_task(run())
        except RuntimeError:
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
