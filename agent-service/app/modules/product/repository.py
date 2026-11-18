import re

from sqlalchemy import *
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.product.models import Product
from decimal import Decimal

# 产品名比对前去掉空格、标点等噪声，避免"小蜜蜂 6 号"与"小蜜蜂6号"匹配不上
_NAME_NOISE = re.compile(r"[\s\-_/（）()【】\[\]·.。，,]")


def _normalize_product_name(name: str) -> str:
    return _NAME_NOISE.sub("", name).lower()


class ProductRepository:

    def __init__(self, session: AsyncSession):
        self.session = session


    async def find_by_name(self, name: str) -> Product | None:
        """按产品名查找上架产品：先精确匹配，再双向包含匹配

        产品表规模很小，直接取全量在内存里比对，兼容"太平洋小蜜蜂6号"
        与库中"小蜜蜂6号"这类带渠道前缀/后缀的写法。
        """

        target = _normalize_product_name(name)
        if not target:
            return None

        products = (
            await self.session.scalars(
                select(Product).where(Product.status == "active")
            )
        ).all()
        if not products:
            return None

        # 1.精确匹配（归一化后）
        for product in products:
            if _normalize_product_name(product.name) == target:
                return product

        # 2.双向包含匹配，命中多个时取名称最长（最具体）的一个
        contained = [
            product
            for product in products
            if target in _normalize_product_name(product.name)
            or _normalize_product_name(product.name) in target
            or target in _normalize_product_name(product.clause_name)
        ]
        if contained:
            return max(contained, key=lambda p: len(p.name))
        return None


    async def query_by_category(
        self,
        category: str | None = None,
    ) -> list[Product]:

        # 1.动态组装查询条件
        # 1.1.产品必须是上架产品，status是active
        conditions = [Product.status == 'active']
        # 1.1.分类条件
        if category:
            conditions.append(Product.category == category)

        # 2.执行查询
        result = await self.session.scalars(
            select(Product)
            .where(*conditions)
            .order_by(Product.id.asc())
        )

        # 3.返回结果
        return result.all()

    async def find_candidates(
        self,
        categories: list[str],
        premium_min: Decimal | None,
        limit_per_category: int,
    ) -> list[Product]:
        # 每个险种查limit_per_category条，所以先定义list，收集结果
        candidate_products: list[Product] = []

        # 利用dict去重，并保留调用方传入的险种顺序，然后遍历逐个查询
        for category in dict.fromkeys(categories):
            conditions = [
                Product.status == "active",
                Product.category == category,
            ]
            if premium_min is not None:
                conditions.append(Product.min_premium < premium_min)

            products = await self.session.scalars(
                select(Product)
                .where(*conditions)
                .order_by(
                    Product.min_premium.asc().nullslast(),
                    Product.id.asc(),
                )
                .limit(limit_per_category)
            )
            candidate_products.extend(products.all())

        return candidate_products