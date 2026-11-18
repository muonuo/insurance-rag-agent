from collections import defaultdict
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.insurance_plan.repository import InsurancePlanRepository
from app.modules.insurance_plan.schemas import InsurancePlanCreate


class InsurancePlanService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = InsurancePlanRepository(session)

    async def create_plan(
        self,
        user_id: int,
        data: InsurancePlanCreate,
    ) -> UUID:
        # 方案和方案项必须作为一个整体保存：全部成功才提交，任何一步失败都回滚
        try:
            plan = await self.repository.create(
                user_id=user_id,
                data=data,
            )
            await self.session.commit()
            return plan.id
        except Exception:
            await self.session.rollback()
            raise

    async def list_plans(self, user_id: int, limit: int = 10) -> list[dict]:
        """查询用户保存过的方案（跨会话），补齐方案项和产品名称

        返回纯 dict（UUID/Decimal/时间原样保留），由调用方做JSON编码。
        """

        # 1.查方案（新的在前）
        plans = await self.repository.list_by_user(user_id, limit)
        if not plans:
            return []

        # 2.批量查方案项
        items = await self.repository.list_items([plan.id for plan in plans])

        # 3.批量查产品名称，避免逐项查询（N+1）
        product_ids = list(dict.fromkeys(item.product_id for item in items))
        products = await self.repository.list_products(product_ids)
        product_names = {product.id: product.name for product in products}

        # 4.按方案归组
        items_by_plan: dict[UUID, list[dict]] = defaultdict(list)
        for item in items:
            items_by_plan[item.plan_id].append(
                {
                    "product_id": item.product_id,
                    "product_name": product_names.get(item.product_id),
                    "priority": item.priority,
                    "recommendation_reason": item.recommendation_reason,
                    "annual_premium_budget": item.annual_premium_budget,
                    "status": item.status,
                }
            )

        return [
            {
                "plan_id": plan.id,
                "plan_name": plan.plan_name,
                "summary": plan.summary,
                "status": plan.status,
                "annual_premium_budget": plan.annual_premium_budget,
                "insured_profile": plan.insured_profile,
                "created_at": plan.created_at,
                "items": items_by_plan.get(plan.id, []),
            }
            for plan in plans
        ]
