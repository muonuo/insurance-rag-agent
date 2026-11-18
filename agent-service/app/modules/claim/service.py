from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.claim.repository import ClaimRepository


class ClaimService:
    def __init__(self, session: AsyncSession):
        self.repository = ClaimRepository(session)

    async def list_policies(self, user_id: int) -> list[dict[str, Any]]:
        return await self.repository.list_policies(user_id)

    async def list_claims(self, user_id: int) -> list[dict[str, Any]]:
        return await self.repository.list_claims(user_id)
