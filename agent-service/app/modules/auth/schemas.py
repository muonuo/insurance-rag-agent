import re

from pydantic import BaseModel, Field, field_validator

from app.common.schemas import BaseSchema

# 用户名规则：登录与注册共用，保持一致
USERNAME_FIELD = Field(min_length=3, max_length=80)
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class LoginRequest(BaseModel):
    """登录请求"""

    username: str = USERNAME_FIELD
    password: str = Field(min_length=6, max_length=64)

    @field_validator("username", "password", mode="before")
    @classmethod
    def strip_spaces(cls, value: str) -> str:
        return value.strip()


class RegisterRequest(LoginRequest):
    """注册请求"""

    email: str = Field(min_length=3, max_length=160)
    display_name: str | None = Field(default=None, max_length=100)

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not EMAIL_PATTERN.match(value):
            raise ValueError("邮箱格式不正确")
        return value

    @field_validator("display_name", mode="before")
    @classmethod
    def normalize_display_name(cls, value: str | None) -> str | None:
        # 输入纯空格时视为未填写
        if value is None:
            return None
        value = value.strip()
        return value or None


class UserResponse(BaseSchema):
    """用户信息响应"""

    id: int
    username: str
    email: str
    display_name: str | None


class AuthResponse(BaseModel):
    """登录/注册成功响应（注册即自动登录）"""

    token: str
    # 令牌有效期（秒）
    expires_in: int
    user: UserResponse
