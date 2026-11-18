"""产品名解析与条款检索工具测试：名称→ID解析、工具路由与明确报错"""

import asyncio
import json
from types import SimpleNamespace

import app.agents.tools.clause as clause_module
from app.agents.tools.clause import retrieve_insurance_clause
from app.modules.product.repository import ProductRepository
from app.rag.capture import capture_contexts


class FakeScalars:
    def __init__(self, items):
        self._items = items

    def all(self):
        return list(self._items)


class FakeSession:
    """假AsyncSession：只提供repository用到的scalars"""

    def __init__(self, products):
        self.products = products

    async def scalars(self, stmt):
        return FakeScalars(self.products)


def product(name: str, clause_name: str = "", pid: int = 1, status: str = "active"):
    return SimpleNamespace(id=pid, name=name, clause_name=clause_name or f"{name}条款", status=status)


class TestFindByName:
    def _repo(self, products):
        return ProductRepository(FakeSession(products))

    def test_exact_match(self):
        repo = self._repo([product("小蜜蜂6号综合意外险", pid=3)])
        assert asyncio.run(repo.find_by_name("小蜜蜂6号综合意外险")).id == 3

    def test_ignores_spaces_and_punctuation(self):
        repo = self._repo([product("小蜜蜂6号综合意外险", pid=3)])
        assert asyncio.run(repo.find_by_name("小蜜蜂 6 号（综合意外险）")).id == 3

    def test_query_longer_than_db_name(self):
        # 用户说全名带渠道前缀，库里是短名
        repo = self._repo([product("小蜜蜂6号综合意外险", pid=3)])
        assert asyncio.run(repo.find_by_name("太平洋小蜜蜂6号综合意外险")).id == 3

    def test_db_name_longer_than_query(self):
        repo = self._repo([product("小蜜蜂6号综合意外险2026版", pid=4)])
        assert asyncio.run(repo.find_by_name("小蜜蜂6号")).id == 4

    def test_matches_clause_name(self):
        repo = self._repo([product("星辰守护B款", clause_name="信美相互星辰守护B款少儿重疾险条款.pdf", pid=1)])
        assert asyncio.run(repo.find_by_name("星辰守护B款少儿重疾险")).id == 1

    def test_not_found(self):
        repo = self._repo([product("小蜜蜂6号综合意外险", pid=3)])
        assert asyncio.run(repo.find_by_name("完全不存在的产品")) is None

    def test_empty_name(self):
        repo = self._repo([product("小蜜蜂6号综合意外险", pid=3)])
        assert asyncio.run(repo.find_by_name("   ")) is None


def _invoke(payload: dict):
    return asyncio.run(retrieve_insurance_clause.ainvoke(payload))


class TestRetrieveTool:
    def setup_method(self):
        self.searched: list[tuple] = []
        # 工具写事件流需要langgraph运行时，测试里替换成空操作
        clause_module.get_stream_writer = lambda: (lambda data: None)

    def _patch(self, monkeypatch, *, resolved=3, sources=None):
        async def fake_resolve(name):
            return resolved

        async def fake_search(query, k, product_id, key_query=None):
            self.searched.append((query, k, product_id, key_query))
            return sources

        monkeypatch.setattr(clause_module, "_resolve_product_id", fake_resolve)
        monkeypatch.setattr(clause_module, "_search_clauses", fake_search)

    def test_resolves_name_then_searches_with_resolved_id(self, monkeypatch):
        sources = {"ref-001": {"product_id": 3, "clause_name": "c", "section_path": [], "content": "条款正文"}}
        self._patch(monkeypatch, resolved=3, sources=sources)

        with capture_contexts() as bucket:
            result = _invoke({"query": "等待期多久", "product_name": "小蜜蜂6号综合意外险"})

        assert json.loads(result)["ref-001"]["content"] == "条款正文"
        # 检索用的是解析后的ID，不是模型猜的
        assert self.searched == [("等待期多久", 5, 3, None)]
        # 评测上下文被记录
        assert bucket == ["条款正文"]

    def test_unknown_product_name_returns_clear_message(self, monkeypatch):
        self._patch(monkeypatch, resolved=None, sources={"ref-001": {"content": "x"}})

        result = _invoke({"query": "等待期多久", "product_name": "不存在的产品"})

        assert "没有找到名为「不存在的产品」" in result
        # 不检索，避免把别的产品条款当成答案
        assert self.searched == []

    def test_no_match_returns_message_without_recording(self, monkeypatch):
        self._patch(monkeypatch, resolved=3, sources=None)

        with capture_contexts() as bucket:
            result = _invoke({"query": "等待期多久", "product_name": "小蜜蜂6号"})

        assert "没有查询到相关保险条款" in result
        assert bucket == []

    def test_product_id_fallback_when_name_empty(self, monkeypatch):
        self._patch(monkeypatch, resolved=99, sources=None)

        _invoke({"query": "等待期", "product_name": "", "product_id": 7})

        # product_name为空时走product_id，不调用名称解析
        assert self.searched == [("等待期", 5, 7, None)]

    def test_neither_name_nor_id(self, monkeypatch):
        self._patch(monkeypatch)

        result = _invoke({"query": "等待期", "product_name": ""})

        assert "请提供要查询条款的保险产品名称" in result
        assert self.searched == []

    def test_key_query_uses_last_user_message(self, monkeypatch):
        # 缓存定位用用户原话，而不是模型生成的query
        from langchain.tools import ToolRuntime
        from langchain_core.messages import HumanMessage

        from app.agents.tools.clause import retrieve_insurance_clause as clause_tool

        self._patch(monkeypatch, resolved=3, sources=None)
        runtime = ToolRuntime(
            state={
                "messages": [
                    HumanMessage("你好"),
                    HumanMessage("看看第一个产品的等待期"),
                ]
            },
            context=SimpleNamespace(user_id=1),
            config={},
            stream_writer=lambda _: None,
            tool_call_id="call-1",
            store=None,
        )

        asyncio.run(
            clause_tool.coroutine(
                query="等待期", product_name="小蜜蜂6号", runtime=runtime
            )
        )

        # 第4个参数是key_query，应为最后一条用户原话
        assert self.searched[0][3] == "看看第一个产品的等待期"
