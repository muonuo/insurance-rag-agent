from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction
from app.core.config import settings
from app.rag.cache import RetrievalCache
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import ParentChunkRetriever

# 1.初始化向量模型
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",  # 性价比高的模型
    dimensions=1024  # 可选：减少维度以节省存储
)

# 2、初始化 VectorStore，会自动创建Collection
vector_store = Milvus(
    embedding_function=embeddings,
    collection_name="insurance_collection",
    builtin_function=BM25BuiltInFunction(analyzer_params={"type": "chinese"}),
    vector_field=["dense", "sparse"],
    connection_args={"uri": settings.rag.milvus_url},
    auto_id=True,
)

# 3.创建检索缓存（惰性连接：导入时不连Redis，首次get/set才连；故障直通检索）
retrieval_cache = RetrievalCache(
    url=settings.rag.cache_redis_url,
    ttl=settings.rag.cache_ttl,
    enabled=settings.rag.cache_enabled,
)

# 4.创建Pipeline（带缓存引用：条款重新入库后按产品失效缓存）
rag_pipeline = RAGPipeline(vector_store, cache=retrieval_cache)

# 5.创建Retriever（带缓存：同样的query直接返回，跳过混合检索+重排+回表）
retriever = ParentChunkRetriever(vector_store, cache=retrieval_cache)


# 6.关闭资源
# 注：Redis客户端的连接随进程退出自动关闭；close_rag保持同步签名不变
def close_rag():
    rag_pipeline.close()
    vector_store.client.close()

__all__ = ['pipeline', 'retriever', 'close_rag']