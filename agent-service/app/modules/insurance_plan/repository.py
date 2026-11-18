from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.insurance_plan.models import (
    InsurancePlan,
    InsurancePlanItem,
)
from app.modules.insurance_plan.schemas import InsurancePlanCreate
from app.modules.product.models import Product


class InsurancePlanRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        user_id: int,
        data: InsurancePlanCreate,
    ) -> InsurancePlan:
        # 1.计算方案年缴总预算
        item_budgets = [
            item.annual_premium_budget
            for item in data.items
            if item.annual_premium_budget is not None
        ]
        annual_premium_budget = (
            sum(item_budgets, Decimal("0"))
            if item_budgets
            else None
        )

        # 2.保存保险方案
        plan = InsurancePlan(
            user_id=user_id,
            plan_name=data.plan_name,
            summary=data.summary,
            insured_profile=data.insured_profile,
            annual_premium_budget=annual_premium_budget,
        )
        self.session.add(plan)
        # flush把INSERT发给数据库拿到方案ID，但不提交事务
        await self.session.flush()

        # 3.保存方案项，关联到同一个方案
        plan_items = [
            InsurancePlanItem(
                plan_id=plan.id,
                product_id=item.product_id,
                priority=item.priority,
                recommendation_reason=item.recommendation_reason,
                annual_premium_budget=item.annual_premium_budget,
            )
            for item in data.items
        ]
        self.session.add_all(plan_items)
        return plan

    async def list_by_user(self, user_id: int, limit: int = 10) -> list[InsurancePlan]:
        """查询用户最近的保险方案（新的在前）"""

        result = await self.session.scalars(
            select(InsurancePlan)
            .where(InsurancePlan.user_id == user_id)
            .order_by(InsurancePlan.created_at.desc())
            .limit(limit)
        )
        return list(result.all())

    async def list_items(self, plan_ids: list[UUID]) -> list[InsurancePlanItem]:
        """批量查询方案项，按方案内优先级排序"""

        if not plan_ids:
            return []
        result = await self.session.scalars(
            select(InsurancePlanItem)
            .where(InsurancePlanItem.plan_id.in_(plan_ids))
            .order_by(InsurancePlanItem.priority.asc())
        )
        return list(result.all())

    async def list_products(self, product_ids: list[int]) -> list[Product]:
        """批量查询产品，用于补齐方案项的产品名称"""

        if not product_ids:
            return []
        result = await self.session.scalars(
            select(Product).where(Product.id.in_(product_ids))
        )
        return list(result.all())
