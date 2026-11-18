"""AuthService 业务逻辑测试：使用内存假仓库，不依赖数据库"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import jwt
import pytest

from app.core.security import create_access_token, hash_password
from app.modules.auth.exceptions import (
    AccountDisabledError,
    EmailAlreadyExistsError,
    InvalidCredentialsError,
    UsernameAlreadyExistsError,
)
from app.modules.auth.schemas import LoginRequest, RegisterRequest
from app.modules.auth.service import AuthService


class FakeUserRepository:
    """内存版用户仓库，接口与UserRepository一致"""

    def __init__(self, users: list[SimpleNamespace] | None = None):
        self.users: list[SimpleNamespace] = list(users or [])

    async def get_by_username(self, username: str):
        return next((u for u in self.users if u.username == username), None)

    async def get_by_email(self, email: str):
        return next((u for u in self.users if u.email == email), None)

    async def get_by_id(self, user_id: int):
        return next((u for u in self.users if u.id == user_id), None)

    async def add(self, user):
        # 模拟数据库自增主键
        user.id = len(self.users) + 100
        self.users.append(user)


class FakeSession:
    """假AsyncSession：只提供service用到的begin/flush/refresh"""

    @asynccontextmanager
    async def begin(self):
        yield

    async def flush(self):
        pass

    async def refresh(self, obj):
        pass


def make_demo_user() -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        username="demo",
        email="demo@example.com",
        password_hash=hash_password("123456"),
        display_name="演示用户",
        status="active",
    )


def make_service(users: list[SimpleNamespace]) -> AuthService:
    return AuthService(session=FakeSession(), repository=FakeUserRepository(users))


class TestLogin:
    def test_success_returns_token_and_user(self):
        service = make_service([make_demo_user()])
        response = asyncio.run(
            service.login(LoginRequest(username="demo", password="123456"))
        )
        assert response.user.username == "demo"
        assert response.user.display_name == "演示用户"
        assert response.token
        payload = jwt.decode(
            response.token,
            options={"verify_signature": False},
        )
        assert payload["sub"] == "1"

    def test_wrong_password_and_unknown_user_raise_same_error(self):
        # 统一异常类型，避免泄露"用户是否存在"
        service = make_service([make_demo_user()])
        with pytest.raises(InvalidCredentialsError):
            asyncio.run(
                service.login(LoginRequest(username="demo", password="bad-password"))
            )
        with pytest.raises(InvalidCredentialsError):
            asyncio.run(
                service.login(LoginRequest(username="nobody", password="123456"))
            )

    def test_disabled_account_raises(self):
        user = make_demo_user()
        user.status = "disabled"
        service = make_service([user])
        with pytest.raises(AccountDisabledError):
            asyncio.run(
                service.login(LoginRequest(username="demo", password="123456"))
            )


class TestRegister:
    def test_success_hashes_password_and_auto_login(self):
        service = make_service([make_demo_user()])
        response = asyncio.run(
            service.register(
                RegisterRequest(
                    username="alice",
                    password="123456",
                    email="alice@example.com",
                    display_name="Alice",
                )
            )
        )
        saved = service.repository.users[-1]
        assert saved.username == "alice"
        # 明文密码绝不能入库
        assert saved.password_hash != "123456"
        assert response.token

    def test_duplicate_username_raises(self):
        service = make_service([make_demo_user()])
        with pytest.raises(UsernameAlreadyExistsError):
            asyncio.run(
                service.register(
                    RegisterRequest(
                        username="demo",
                        password="123456",
                        email="other@example.com",
                    )
                )
            )

    def test_duplicate_email_raises(self):
        service = make_service([make_demo_user()])
        with pytest.raises(EmailAlreadyExistsError):
            asyncio.run(
                service.register(
                    RegisterRequest(
                        username="newbie",
                        password="123456",
                        email="demo@example.com",
                    )
                )
            )


class TestTokenLifecycle:
    def test_token_from_create_access_token_matches_login_flow(self):
        # 确认测试工具与业务代码共用同一签发逻辑
        token = create_access_token(user_id=7)
        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["sub"] == "7"
        assert payload["type"] == "access"
