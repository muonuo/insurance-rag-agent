from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.core.security import (
    DUMMY_PASSWORD_HASH,
    create_access_token,
    hash_password,
    verify_password,
)
from app.core.config import settings

from .exceptions import (
    AccountDisabledError,
    EmailAlreadyExistsError,
    InvalidCredentialsError,
    UnauthorizedError,
    UsernameAlreadyExistsError,
)
from .models import User
from .repository import UserRepository
from .schemas import AuthResponse, LoginRequest, RegisterRequest, UserResponse


class AuthService:

    def __init__(self, session: AsyncSession, repository: UserRepository | None = None):
        self.session = session
        # 支持注入自定义repository，便于单元测试
        self.repository = repository or UserRepository(session)

    async def register(self, request: RegisterRequest) -> AuthResponse:
        # 1.创建用户对象，明文密码绝不能入库
        # 哈希计算较耗时（argon2约百毫秒），放到线程池避免阻塞事件循环
        password_hash = await run_in_threadpool(hash_password, request.password)
        user = User(
            username=request.username,
            email=request.email,
            password_hash=password_hash,
            display_name=request.display_name,
            status="active",
        )

        # 2.事务内查重并写入，成功后自动commit，出现异常自动rollback
        # 注意：查询必须在begin()内部。SQLAlchemy的Session在首次查询时会自动开启事务，
        # 若先查询再进入begin()，会抛出"A transaction is already begun on this Session"
        try:
            async with self.session.begin():
                if await self.repository.get_by_username(request.username):
                    raise UsernameAlreadyExistsError()
                if await self.repository.get_by_email(request.email):
                    raise EmailAlreadyExistsError()

                await self.repository.add(user)
                await self.session.flush()
                await self.session.refresh(user)
        except IntegrityError as e:
            # 并发注册时预检查可能被同时穿过，最终由数据库唯一约束兜底
            detail = str(e.orig)
            if "username" in detail:
                raise UsernameAlreadyExistsError() from e
            if "email" in detail:
                raise EmailAlreadyExistsError() from e
            raise

        # 3.注册即自动登录，直接签发令牌
        return self._auth_response(user)

    async def login(self, request: LoginRequest) -> AuthResponse:
        # 1.查询用户
        user = await self.repository.get_by_username(request.username)
        # 2.用户不存在时也执行一次哈希校验，保证耗时一致，避免通过响应时间枚举用户名
        if user is None:
            await run_in_threadpool(verify_password, request.password, DUMMY_PASSWORD_HASH)
            raise InvalidCredentialsError()
        # 3.密码错误同样抛这个异常，避免泄露"用户是否存在"
        if not await run_in_threadpool(verify_password, request.password, user.password_hash):
            raise InvalidCredentialsError()
        # 4.校验账户状态
        if user.status != "active":
            raise AccountDisabledError()

        return self._auth_response(user)

    async def get_user(self, user_id: int) -> User:
        """根据ID查询用户（用于令牌校验后确认用户仍然存在）"""
        user = await self.repository.get_by_id(user_id)
        if user is None:
            # 令牌还有效，但用户已被删除
            raise UnauthorizedError()
        return user

    def _auth_response(self, user: User) -> AuthResponse:
        """封装登录/注册的成功响应"""
        return AuthResponse(
            token=create_access_token(user_id=user.id),
            expires_in=settings.security.jwt_expires_hours * 3600,
            user=UserResponse.model_validate(user),
        )
