"""app.rag.retriever.rerank_documents 测试：正常排序与API错误回退"""

import sys
from types import SimpleNamespace

from langchain_core.documents import Document

from app.rag.retriever import rerank_documents

# 注意：不能用 `import app.rag.retriever as m`——app/rag/__init__.py 里的
# `retriever = ParentChunkRetriever(...)` 会遮蔽同名子模块属性，拿到的是实例
retriever_module = sys.modules["app.rag.retriever"]


def make_docs(*texts):
    return [Document(page_content=t) for t in texts]


class TestRerankDocuments:
    def test_empty_docs_returns_empty(self):
        assert rerank_documents("问题", [], top_k=3) == []

    def test_orders_by_relevance_score(self, monkeypatch):
        docs = make_docs("不相关", "最相关", "中等")
        fake = SimpleNamespace(
            status_code=200,
            code="",
            message="",
            output=SimpleNamespace(
                results=[
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 2, "relevance_score": 0.5},
                    {"index": 0, "relevance_score": 0.01},
                ]
            ),
        )
        monkeypatch.setattr(retriever_module.TextReRank, "call", lambda **kw: fake)

        result = rerank_documents("问题", docs, top_k=2)

        assert [d.page_content for d in result] == ["最相关", "中等"]

    def test_drops_non_positive_scores(self, monkeypatch):
        docs = make_docs("相关", "为零")
        fake = SimpleNamespace(
            status_code=200,
            code="",
            message="",
            output=SimpleNamespace(
                results=[
                    {"index": 0, "relevance_score": 0.8},
                    {"index": 1, "relevance_score": 0.0},
                ]
            ),
        )
        monkeypatch.setattr(retriever_module.TextReRank, "call", lambda **kw: fake)

        result = rerank_documents("问题", docs, top_k=5)

        # 0分视为不相关，直接丢弃
        assert [d.page_content for d in result] == ["相关"]

    def test_api_error_falls_back_to_original_order(self, monkeypatch):
        # 鉴权失败/模型不存在时SDK返回错误码对象而不是抛异常：
        # 必须回退到RRF顺序，绝不能当成"无结果"（否则会缓存空结果）
        docs = make_docs("A", "B", "C")
        fake = SimpleNamespace(
            status_code=403,
            code="AccessDenied",
            message="Access denied",
            output=None,
        )
        monkeypatch.setattr(retriever_module.TextReRank, "call", lambda **kw: fake)

        result = rerank_documents("问题", docs, top_k=2)

        assert [d.page_content for d in result] == ["A", "B"]

    def test_exception_falls_back_to_original_order(self, monkeypatch):
        docs = make_docs("A", "B")

        def boom(**kwargs):
            raise RuntimeError("network down")

        monkeypatch.setattr(retriever_module.TextReRank, "call", boom)

        result = rerank_documents("问题", docs, top_k=2)

        assert [d.page_content for d in result] == ["A", "B"]
