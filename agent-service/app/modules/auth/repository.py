from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import User


class UserRepository:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def add(self, user: User):
        self.session.add(user)

    async def get_by_username(self, username: str) -> User | None:
        return await self.session.scalar(
            # SELECT * FROM users WHERE username = 'demo'
            select(User).where(User.username == username)
        )

    async def get_by_email(self, email: str) -> User | None:
        return await self.session.scalar(
            select(User).where(User.email == email)
        )

    async def get_by_id(self, user_id: int) -> User | None:
        return await self.session.scalar(
            select(User).where(User.id == user_id)
        )
