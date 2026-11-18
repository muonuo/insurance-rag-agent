import asyncio
import time

from dashscope import TextReRank
from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import Function, FunctionType

from app.core.config import settings
from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory
from .repository import ParentChunkRepository

logger = get_logger(__name__)

# RRF融合后进入精排的候选数 = max(k * 倍数, 下限)
RRF_CANDIDATE_FACTOR = 3
RRF_MIN_CANDIDATES = 15
# 重排得分不高于该值视为不相关，直接丢弃
MIN_RERANK_SCORE = 0.0


def create_rrf_ranker() -> Function:
    """创建RRF融合器，按名次投票融合稠密和稀疏两路召回结果。

    两路分数量纲不可比（余弦相似度 vs BM25词频分），但名次天然可比：
    两路都靠前的文档，得分必然高于只有一路靠前的文档。
    """
    return Function(
        name="rrf_ranker",
        input_field_names=[],  # RRF Ranker要求必须为空列表
        function_type=FunctionType.RERANK,
        params={"reranker": "rrf", "k": settings.rag.rrf_k},
    )


def rerank_documents(query: str, docs: list[Document], top_k: int) -> list[Document]:
    """第二阶段：调用云端重排模型，对RRF融合后的候选做逐对精排（同步调用）"""

    if not docs:
        return []

    started = time.perf_counter()
    try:
        response = TextReRank.call(
            api_key=settings.llm.dashscope_api_key,
            model=settings.rag.rerank_model,
            query=query,
            documents=[doc.page_content for doc in docs],
            top_n=top_k,
            return_documents=False,
        )
        # DashScope SDK在鉴权失败/模型不存在时不会抛异常，而是返回错误码对象；
        # 不检查会把"API失败"误判成"检索无结果"，还会把空结果写进缓存
        if getattr(response, "status_code", 200) != 200:
            logger.warning(
                "条款重排返回错误，回退到RRF融合顺序",
                model=settings.rag.rerank_model,
                status=getattr(response, "status_code", None),
                code=getattr(response, "code", None),
                message=str(getattr(response, "message", ""))[:120],
                candidates=len(docs),
            )
            return docs[:top_k]
        results = response.output.results if response.output else []
        # 按相关度降序，分数为0的视为不相关直接丢弃
        ranked = sorted(
            results, key=lambda item: item["relevance_score"], reverse=True
        )
        kept = [
            docs[item["index"]]
            for item in ranked
            if item["relevance_score"] > MIN_RERANK_SCORE
        ][:top_k]
        logger.info(
            "条款重排完成",
            model=settings.rag.rerank_model,
            candidates=len(docs),
            kept=len(kept),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return kept
    except Exception:
        # 重排只是增强手段，失败时回退到RRF顺序，不影响主流程
        logger.exception(
            "条款重排失败，回退到RRF融合顺序",
            model=settings.rag.rerank_model,
            candidates=len(docs),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return docs[:top_k]


class ParentChunkRetriever:

    def __init__(self, vector_store: Milvus, cache=None):
        self._vector_store = vector_store
        self._cache = cache  # RetrievalCache | None：None时直通检索，行为与无缓存一致

    async def retrieve(
        self,
        query: str,
        k: int,
        product_id: int,
        key_query: str | None = None,
    ):
        """检索条款。key_query用于缓存定位（通常传用户原话，比模型生成的query稳定）"""

        started = time.perf_counter()

        # 0.查L1缓存：key由归一化后的key_query+product_id+k决定，命中直接返回
        cache_hit = False
        if self._cache is not None:
            cached = await self._cache.get(query, k, product_id, key_query=key_query)
            if cached is not None:
                cache_hit = True
                self._cache.record_hit()
                logger.info(
                    "条款检索完成",
                    product_id=product_id,
                    k=k,
                    query=query[:50],
                    key_query=(key_query or "")[:50] or None,
                    parents=len(cached),
                    cache_hit=True,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )
                return cached
            self._cache.record_miss()

        # 1.混合检索：稠密+稀疏两路召回，用RRF按名次融合
        #    先多取候选，留给第二阶段精排
        candidate_k = max(k * RRF_CANDIDATE_FACTOR, RRF_MIN_CANDIDATES)
        hybrid_started = time.perf_counter()
        child_chunks: list[Document] = await self._vector_store.asimilarity_search(
            query,
            k=candidate_k,       # RRF融合后返回的候选数量
            fetch_k=candidate_k,  # 每路初筛数量
            expr=f"product_id == {product_id}",
            reranker=create_rrf_ranker(),
        )
        hybrid_ms = (time.perf_counter() - hybrid_started) * 1000

        # 2.模型精排后截取top-k
        #    同步HTTP调用，放线程池避免阻塞事件循环
        reranked = await asyncio.to_thread(
            rerank_documents, query, child_chunks, k
        )
        if not reranked:
            logger.info(
                "条款检索无结果",
                product_id=product_id,
                k=k,
                query=query[:50],
                candidates=len(child_chunks),
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            # 无结果也写缓存，避免同一问题反复触发完整检索
            if self._cache is not None:
                await self._cache.set(query, k, product_id, [], key_query=key_query)
            return []

        # 3.从子块中拿到parent_id，并且去重
        parent_ids = list(
            dict.fromkeys(
                child.metadata.get("parent_id") for child in reranked
            )
        )
        if not parent_ids:
            return []

        # 4.查询父块
        async with AsyncSessionFactory() as session:
            repo = ParentChunkRepository(session)
            parents = await repo.query_by_ids(parent_ids)

        # 5.把parents按照检索回来的顺序排序
        parent_id_dict = {str(parent.id): parent for parent in parents}
        parent_chunks = [
            parent_id_dict.get(parent_id) for parent_id in parent_ids
        ]

        # 过滤掉父块缺失的情况，避免调用方拿到None
        results = [chunk for chunk in parent_chunks if chunk is not None]

        # 6.写回缓存（含空结果：防止无结果的问题反复触发完整检索）
        if self._cache is not None:
            await self._cache.set(query, k, product_id, results, key_query=key_query)

        # 7.检索链路可观测性：一次性输出各阶段耗时与命中数，
        #    用于区分"检索慢"还是"检索不准"
        logger.info(
            "条款检索完成",
            product_id=product_id,
            k=k,
            query=query[:50],
            key_query=(key_query or "")[:50] or None,
            candidates=len(child_chunks),      # RRF融合后的候选数
            reranked=len(reranked),            # 重排保留的子块数
            parents=len(results),              # 最终返回的父块数
            hybrid_ms=round(hybrid_ms, 1),     # 混合检索耗时
            cache_hit=cache_hit,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return results
