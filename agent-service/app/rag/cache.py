"""RAG检索结果L1缓存：Redis精确匹配（同样的query直接返回，跳过检索链路）

只缓存"问题 → 检索到的父块"，不缓存LLM回答。
- Key = rag:retrieval:{product_id}:{sha256(query|k)}，product_id放外层支持按产品整批失效
- Value = JSON序列化的父块字段（不用pickle：SQLAlchemy对象序列化脆、不可读、跨版本危险）
- fail-open：Redis不可用时告警并直通检索，绝不影响对话
- 惰性连接：构造时不连网，首次get/set才创建客户端，保证导入与测试零依赖
"""

import hashlib
import json
import re
from uuid import UUID

from app.core.logging import get_logger
from .models import ParentChunk

logger = get_logger(__name__)

# 命中率统计的输出间隔（次）
STATS_EVERY = 50

_KEY_PREFIX = "rag:retrieval"

# key归一化时要去掉的标点与空白（中英文都覆盖）
_KEY_NOISE = re.compile(r"[\s、，,。.;；:：!！?？\"'“”‘’()（）\[\]【】<>《》\-_/\\|]+")


def normalize_query(query: str) -> str:
    """归一化查询用于计算缓存key：去标点空格、转小写、分段排序去重

    只影响key，不影响真正发给检索引擎的query。
    "等待期、免赔额" 与 "免赔额 等待期" 归一化后相同，可命中同一条缓存。
    """

    parts = [part for part in _KEY_NOISE.split(query.lower()) if part]
    if not parts:
        return ""
    return "|".join(sorted(set(parts)))


def chunk_to_dict(chunk: ParentChunk) -> dict:
    """父块 → 可JSON序列化的dict（显式字段，避免ORM对象的隐式状态）"""

    return {
        "id": str(chunk.id),
        "product_id": chunk.product_id,
        "clause_name": chunk.clause_name,
        "section_path": list(chunk.section_path or []),
        "content": chunk.content,
    }


def chunk_from_dict(data: dict) -> ParentChunk:
    """dict → 重建脱离session的ParentChunk实例（调用方只读属性，够用）"""

    return ParentChunk(
        id=UUID(data["id"]),
        product_id=data["product_id"],
        clause_name=data["clause_name"],
        section_path=list(data.get("section_path") or []),
        content=data["content"],
    )


class RetrievalCache:
    """检索结果缓存：get命中返回重建的父块列表，未命中返回None由调用方走完整检索"""

    def __init__(self, url: str, ttl: int, enabled: bool = True):
        self._url = url
        self._ttl = ttl
        self.enabled = enabled
        self._client = None  # 惰性创建
        # 命中率统计（进程内）：定期输出汇总，观察命中率再决定是否值得上L2语义缓存
        self._hits = 0
        self._misses = 0

    # ---------- 内部 ----------

    def _get_client(self):
        if self._client is None:
            import redis.asyncio as redis

            self._client = redis.Redis.from_url(
                self._url,
                decode_responses=False,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
        return self._client

    def _key(
        self, query: str, k: int, product_id: int, key_query: str | None = None
    ) -> str:
        # key_query用于稳定定位（通常是用户原话），query只作为兜底
        normalized = normalize_query(key_query or query)
        digest = hashlib.sha256(f"{normalized}|{k}".encode("utf-8")).hexdigest()
        return f"{_KEY_PREFIX}:{product_id}:{digest}"

    # ---------- 对外 ----------

    async def get(
        self,
        query: str,
        k: int,
        product_id: int,
        key_query: str | None = None,
    ) -> list[ParentChunk] | None:
        """查缓存；命中返回父块列表，未命中/禁用/故障返回None"""

        if not self.enabled:
            return None
        try:
            raw = await self._get_client().get(self._key(query, k, product_id, key_query))
        except Exception as e:
            logger.warning("检索缓存不可用，直通检索", error=str(e)[:120])
            return None
        if raw is None:
            return None
        chunks = [chunk_from_dict(item) for item in json.loads(raw)]
        return chunks

    async def set(
        self,
        query: str,
        k: int,
        product_id: int,
        chunks: list[ParentChunk],
        key_query: str | None = None,
    ) -> None:
        """写缓存；失败只告警，不影响检索结果返回"""

        if not self.enabled:
            return
        try:
            payload = json.dumps(
                [chunk_to_dict(chunk) for chunk in chunks], ensure_ascii=False
            ).encode("utf-8")
            await self._get_client().setex(
                self._key(query, k, product_id, key_query), self._ttl, payload
            )
        except Exception as e:
            logger.warning("检索缓存写入失败", error=str(e)[:120])

    async def invalidate_product(self, product_id: int) -> int:
        """条款重新入库后按产品清缓存；返回删除的key数"""

        if not self.enabled:
            return 0
        deleted = 0
        try:
            client = self._get_client()
            pattern = f"{_KEY_PREFIX}:{product_id}:*"
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor=cursor, match=pattern, count=500)
                if keys:
                    deleted += await client.delete(*keys)
                if cursor == 0:
                    break
            if deleted:
                logger.info("检索缓存已按产品失效", product_id=product_id, deleted=deleted)
        except Exception as e:
            logger.warning("检索缓存失效失败（TTL会兜底）", product_id=product_id, error=str(e)[:120])
        return deleted

    # ---------- 命中率统计 ----------

    def record_hit(self) -> None:
        self._hits += 1
        self._maybe_log_stats()

    def record_miss(self) -> None:
        self._misses += 1
        self._maybe_log_stats()

    def _maybe_log_stats(self) -> None:
        total = self._hits + self._misses
        if total == 0 or total % STATS_EVERY != 0:
            return
        rate = self._hits / total * 100
        logger.info(
            "检索缓存命中率",
            hits=self._hits,
            misses=self._misses,
            total=total,
            hit_rate=f"{rate:.1f}%",
        )
