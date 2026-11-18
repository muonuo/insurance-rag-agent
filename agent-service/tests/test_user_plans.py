"""用户保险方案查询测试：跨会话方案聚合、产品名补齐与工具输出"""

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import app.agents.tools.plan as plan_tool_module
from app.agents.tools.plan import query_user_plans
from app.modules.insurance_plan.service import InsurancePlanService


def plan(plan_name="家庭保障方案", status="uninsured", budget=Decimal("8000")):
    return SimpleNamespace(
        id=uuid4(),
        plan_name=plan_name,
        summary="覆盖医疗+重疾",
        status=status,
        annual_premium_budget=budget,
        insured_profile={"age": 30},
        created_at=datetime(2026, 11, 17, 10, 0, 0),
    )


def item(plan_id, product_id, priority=1, reason="补充住院保障"):
    return SimpleNamespace(
        plan_id=plan_id,
        product_id=product_id,
        priority=priority,
        recommendation_reason=reason,
        annual_premium_budget=Decimal("3000"),
        status="uninsured",
    )


class FakePlanRepository:
    def __init__(self, plans, items, products):
        self._plans = plans
        self._items = items
        self._products = products

    async def list_by_user(self, user_id, limit=10):
        return self._plans

    async def list_items(self, plan_ids):
        return self._items

    async def list_products(self, product_ids):
        return self._products


def make_service(plans, items=None, products=None) -> InsurancePlanService:
    service = InsurancePlanService(session=None)
    service.repository = FakePlanRepository(plans, items or [], products or [])
    return service


class TestListPlans:
    def test_groups_items_and_fills_product_names(self):
        p1 = plan("方案A", budget=Decimal("5000"))
        p2 = plan("方案B", budget=Decimal("9000"))
        items = [
            item(p1.id, 1, priority=1),
            item(p1.id, 2, priority=2, reason="重疾兜底"),
            item(p2.id, 1, priority=1),
        ]
        products = [SimpleNamespace(id=1, name="百万医疗险"), SimpleNamespace(id=2, name="重疾险")]
        service = make_service([p1, p2], items, products)

        result = asyncio.run(service.list_plans(user_id=1))

        assert [p["plan_name"] for p in result] == ["方案A", "方案B"]
        assert len(result[0]["items"]) == 2
        assert result[0]["items"][0]["product_name"] == "百万医疗险"
        assert result[0]["items"][1]["product_name"] == "重疾险"
        assert result[1]["items"][0]["product_name"] == "百万医疗险"
        # 预算是Decimal原样保留，由调用方编码
        assert result[0]["annual_premium_budget"] == Decimal("5000")

    def test_plan_without_items(self):
        service = make_service([plan("空方案")])
        result = asyncio.run(service.list_plans(user_id=1))
        assert result[0]["items"] == []

    def test_no_plans_returns_empty(self):
        service = make_service([])
        assert asyncio.run(service.list_plans(user_id=1)) == []

    def test_missing_product_does_not_crash(self):
        p = plan()
        items = [item(p.id, 999)]
        # 产品被下架/删除时名称补不齐，但方案仍应返回
        service = make_service([p], items, products=[])
        result = asyncio.run(service.list_plans(user_id=1))
        assert result[0]["items"][0]["product_name"] is None


class TestQueryUserPlansTool:
    def _invoke(self, user_id=7):
        runtime = SimpleNamespace(context=SimpleNamespace(user_id=user_id))
        return asyncio.run(query_user_plans.coroutine(runtime=runtime))

    def test_returns_plans_json_with_user_from_context(self, monkeypatch):
        captured: dict = {}

        async def fake_fetch(user_id):
            captured["user_id"] = user_id
            p = plan()
            return [
                {
                    "plan_id": p.id,
                    "plan_name": p.plan_name,
                    "summary": p.summary,
                    "status": p.status,
                    "annual_premium_budget": p.annual_premium_budget,
                    "insured_profile": p.insured_profile,
                    "created_at": p.created_at,
                    "items": [
                        {
                            "product_id": 1,
                            "product_name": "百万医疗险",
                            "priority": 1,
                            "recommendation_reason": "补充住院保障",
                            "annual_premium_budget": Decimal("3000"),
                            "status": "uninsured",
                        }
                    ],
                }
            ]

        monkeypatch.setattr(plan_tool_module, "_fetch_plans", fake_fetch)

        output = self._invoke(user_id=42)

        # user_id来自运行时上下文，不接受模型传参
        assert captured["user_id"] == 42
        # 可被JSON序列化（UUID/Decimal/时间都被编码）
        data = json.loads(output)
        assert data[0]["plan_name"] == "家庭保障方案"
        assert data[0]["items"][0]["product_name"] == "百万医疗险"

    def test_no_plans_returns_hint(self, monkeypatch):
        async def fake_fetch(user_id):
            return []

        monkeypatch.setattr(plan_tool_module, "_fetch_plans", fake_fetch)

        assert "还没有保存过保险方案" in self._invoke()
