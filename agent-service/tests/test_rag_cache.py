"""app.rag.cache 检索缓存测试：key唯一性、序列化回环、fail-open、按产品失效"""

import asyncio
import json
from uuid import uuid4

from app.rag.cache import (
    RetrievalCache,
    chunk_from_dict,
    chunk_to_dict,
    normalize_query,
)
from app.rag.models import ParentChunk


class FakeRedis:
    """内存版redis客户端：实现cache用到的get/setex/scan/delete语义"""

    def __init__(self, fail: bool = False):
        self.data: dict[str, bytes] = {}
        self.ttls: dict[str, int] = {}
        self.fail = fail

    def _check(self):
        if self.fail:
            raise RuntimeError("redis down")

    async def get(self, key: str):
        self._check()
        return self.data.get(key)

    async def setex(self, key: str, ttl: int, value: bytes):
        self._check()
        self.data[key] = value
        self.ttls[key] = ttl

    async def scan(self, cursor: int = 0, match: str = "*", count: int = 100):
        self._check()
        import fnmatch

        keys = [k for k in self.data if fnmatch.fnmatch(k, match)]
        return 0, keys

    async def delete(self, *keys: str):
        self._check()
        deleted = sum(1 for k in keys if k in self.data)
        for k in keys:
            self.data.pop(k, None)
            self.ttls.pop(k, None)
        return deleted


def make_cache(client: FakeRedis) -> RetrievalCache:
    cache = RetrievalCache(url="redis://fake:6379/0", ttl=3600, enabled=True)
    cache._client = client  # 注入假客户端，避免真实连接
    return cache


def make_chunk(product_id: int = 1, content: str = "免赔额为1万元") -> ParentChunk:
    return ParentChunk(
        id=uuid4(),
        product_id=product_id,
        clause_name="测试条款.pdf",
        section_path=["1. 我们保什么", "1.5 保险责任"],
        content=content,
    )


class TestNormalizeQuery:
    def test_word_order_and_punctuation_ignored(self):
        # 词序、标点差异归一化后相同
        assert normalize_query("等待期、免赔额") == normalize_query("免赔额 等待期")
        assert normalize_query("等待期，免赔额。") == normalize_query("免赔额、等待期")

    def test_case_insensitive(self):
        assert normalize_query("ABC def") == normalize_query("def abc")

    def test_different_terms_differ(self):
        assert normalize_query("等待期") != normalize_query("免赔额")

    def test_empty_query(self):
        assert normalize_query(" ，。！") == ""


class TestKeyQuery:
    def test_same_key_query_hits_even_if_query_differs(self):
        # 模型两轮生成的query不同，但用户原话相同 → 命中同一缓存
        client = FakeRedis()
        cache = make_cache(client)

        asyncio.run(cache.set("模型query A", 5, 1, [make_chunk()], key_query="看看第一个"))
        hits = asyncio.run(cache.get("模型query B", 5, 1, key_query="看看第一个"))

        assert hits is not None and hits[0].content == "免赔额为1万元"

    def test_key_query_normalized_for_hit(self):
        client = FakeRedis()
        cache = make_cache(client)

        asyncio.run(cache.set("q", 5, 1, [make_chunk()], key_query="等待期、免赔额"))
        # 词序/标点不同仍命中
        assert asyncio.run(cache.get("q2", 5, 1, key_query="免赔额 等待期")) is not None

    def test_without_key_query_falls_back_to_query(self):
        client = FakeRedis()
        cache = make_cache(client)
        asyncio.run(cache.set("等待期", 5, 1, [make_chunk()]))

        assert asyncio.run(cache.get("等待期", 5, 1)) is not None
        assert asyncio.run(cache.get("免赔额", 5, 1)) is None


class TestSerialization:
    def test_roundtrip(self):
        chunk = make_chunk()

        data = chunk_to_dict(chunk)
        # 可JSON序列化（值是纯类型）
        encoded = json.dumps(data, ensure_ascii=False)
        restored = chunk_from_dict(json.loads(encoded))

        assert str(restored.id) == str(chunk.id)
        assert restored.product_id == chunk.product_id
        assert restored.clause_name == chunk.clause_name
        assert restored.section_path == chunk.section_path
        assert restored.content == chunk.content


class TestCacheGetSet:
    def test_miss_then_hit(self):
        client = FakeRedis()
        cache = make_cache(client)
        chunk = make_chunk()

        # 未命中
        assert asyncio.run(cache.get("等待期多久", 5, 1)) is None
        # 写入后命中
        asyncio.run(cache.set("等待期多久", 5, 1, [chunk]))
        hits = asyncio.run(cache.get("等待期多久", 5, 1))

        assert hits is not None and len(hits) == 1
        assert hits[0].content == "免赔额为1万元"

    def test_key_isolated_by_query_k_product(self):
        client = FakeRedis()
        cache = make_cache(client)
        asyncio.run(cache.set("等待期多久", 5, 1, [make_chunk(content="A")]))
        asyncio.run(cache.set("等待期多久", 3, 1, [make_chunk(content="B")]))
        asyncio.run(cache.set("等待期多久", 5, 2, [make_chunk(content="C")]))

        assert asyncio.run(cache.get("等待期多久", 5, 1))[0].content == "A"
        assert asyncio.run(cache.get("等待期多久", 3, 1))[0].content == "B"
        assert asyncio.run(cache.get("等待期多久", 5, 2))[0].content == "C"
        # 不同query不串
        assert asyncio.run(cache.get("免赔额多少", 5, 1)) is None

    def test_empty_result_is_cached(self):
        # 空结果也缓存，防止无结果的问题反复触发完整检索
        client = FakeRedis()
        cache = make_cache(client)

        asyncio.run(cache.set("查不到的问题", 5, 1, []))
        assert asyncio.run(cache.get("查不到的问题", 5, 1)) == []

    def test_ttl_applied(self):
        client = FakeRedis()
        cache = make_cache(client)

        asyncio.run(cache.set("q", 5, 1, [make_chunk()]))
        key = next(iter(client.data))
        assert client.ttls[key] == 3600


class TestFailOpen:
    def test_get_failure_returns_none(self):
        cache = make_cache(FakeRedis(fail=True))
        # Redis挂了不抛异常，直通检索
        assert asyncio.run(cache.get("q", 5, 1)) is None

    def test_set_failure_does_not_raise(self):
        cache = make_cache(FakeRedis(fail=True))
        asyncio.run(cache.set("q", 5, 1, [make_chunk()]))

    def test_disabled_is_noop(self):
        client = FakeRedis()
        cache = RetrievalCache(url="redis://fake", ttl=60, enabled=False)
        cache._client = client

        assert asyncio.run(cache.get("q", 5, 1)) is None
        asyncio.run(cache.set("q", 5, 1, [make_chunk()]))
        assert client.data == {}


class TestInvalidate:
    def test_only_target_product_deleted(self):
        client = FakeRedis()
        cache = make_cache(client)
        asyncio.run(cache.set("等待期", 5, 1, [make_chunk(product_id=1)]))
        asyncio.run(cache.set("免赔额", 5, 1, [make_chunk(product_id=1)]))
        asyncio.run(cache.set("等待期", 5, 2, [make_chunk(product_id=2)]))

        deleted = asyncio.run(cache.invalidate_product(1))

        assert deleted == 2
        assert asyncio.run(cache.get("等待期", 5, 1)) is None
        # 产品2不受影响
        assert asyncio.run(cache.get("等待期", 5, 2)) is not None

    def test_failure_returns_zero(self):
        cache = make_cache(FakeRedis(fail=True))
        assert asyncio.run(cache.invalidate_product(1)) == 0


class TestStats:
    def test_hit_miss_counters(self):
        cache = make_cache(FakeRedis())
        cache.record_miss()
        cache.record_miss()
        cache.record_hit()

        assert cache._hits == 1
        assert cache._misses == 2
