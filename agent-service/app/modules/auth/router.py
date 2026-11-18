from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session

from .dependencies import get_current_user_id
from .schemas import AuthResponse, LoginRequest, RegisterRequest, UserResponse
from .service import AuthService

router = APIRouter(prefix="/api/v1/auth", tags=["认证接口"])


def get_auth_service(session: AsyncSession = Depends(get_session)) -> AuthService:
    return AuthService(session)


@router.post("/register", response_model=AuthResponse)
async def register(
    request: RegisterRequest,
    service: AuthService = Depends(get_auth_service),
) -> AuthResponse:
    """注册新用户（成功即自动登录，返回令牌）"""

    return await service.register(request)


@router.post("/login", response_model=AuthResponse)
async def login(
    request: LoginRequest,
    service: AuthService = Depends(get_auth_service),
) -> AuthResponse:
    """用户名密码登录"""

    return await service.login(request)


@router.get("/me", response_model=UserResponse)
async def me(
    user_id: Annotated[int, Depends(get_current_user_id)],
    service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """查询当前登录用户信息（可用于校验缓存的令牌是否有效）"""

    user = await service.get_user(user_id)
    return UserResponse.model_validate(user)
