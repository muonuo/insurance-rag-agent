from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.claim.models import Claim, Policy
from app.modules.product.models import Product


class ClaimRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_policies(self, user_id: int) -> list[dict[str, Any]]:
        """查询用户所有保单，附带产品信息，按生效时间倒序"""
        rows = (
            await self.session.execute(
                select(Policy, Product)
                .join(Product, Policy.product_id == Product.id)
                .where(Policy.user_id == user_id)
                .order_by(Policy.effective_at.desc())
            )
        ).all()

        return [
            {
                "id": str(policy.id),
                "policy_number": policy.policy_number,
                "status": policy.status,
                "effective_at": _datetime(policy.effective_at),
                "expires_at": _datetime(policy.expires_at),
                "insured_name": policy.insured_name,
                "insured_phone": policy.insured_phone,
                "coverage_amount": _decimal(policy.coverage_amount),
                "premium_amount": _decimal(policy.premium_amount),
                "policy_snapshot": policy.policy_snapshot,
                "coverage_snapshot": policy.coverage_snapshot,
                "product": {
                    "id": product.id,
                    "name": product.name,
                    "category": product.category,
                    "insurer": product.insurer,
                },
            }
            for policy, product in rows
        ]

    async def list_claims(self, user_id: int) -> list[dict[str, Any]]:
        """查询用户所有理赔案件，附带保单号和产品信息，按提交时间倒序"""
        rows = (
            await self.session.execute(
                select(Claim, Policy, Product)
                .join(Policy, Claim.policy_id == Policy.id, isouter=True)
                .join(Product, Claim.product_id == Product.id, isouter=True)
                .where(Claim.user_id == user_id)
                .order_by(Claim.submitted_at.desc())
            )
        ).all()

        return [
            {
                "id": str(claim.id),
                "claim_number": claim.claim_number,
                "claim_type": claim.claim_type,
                "amount": _decimal(claim.amount),
                "description": claim.description,
                "status": claim.status,
                "submitted_at": _datetime(claim.submitted_at),
                "policy_number": policy.policy_number if policy else None,
                "product": {
                    "name": product.name,
                    "category": product.category,
                    "insurer": product.insurer,
                }
                if product
                else None,
            }
            for claim, policy, product in rows
        ]


def _datetime(value: datetime | None) -> str | None:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value else None


def _decimal(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None
