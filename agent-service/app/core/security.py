"""安全相关工具：密码哈希与JWT令牌"""

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError

from app.core.config import settings

# 密码哈希器：推荐配置使用argon2进行哈希，同时支持验证argon2和bcrypt两种格式
password_hasher = PasswordHash.recommended()

# 用户不存在时也校验一次，保证响应耗时一致，防止通过时间差枚举用户名
DUMMY_PASSWORD_HASH = password_hasher.hash("timing-attack-placeholder")


def hash_password(plain: str) -> str:
    """将明文密码哈希为可存储的密文"""
    return password_hasher.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """校验明文密码与存储的哈希是否匹配"""
    try:
        return password_hasher.verify(plain, hashed)
    except (ValueError, UnknownHashError):
        # 存储的哈希格式非法（例如历史占位数据）时，视为校验失败
        return False


def create_access_token(*, user_id: int, expires_delta: timedelta | None = None) -> str:
    """为指定用户签发JWT访问令牌"""
    now = datetime.now(timezone.utc)
    payload = {
        # JWT标准要求sub必须是字符串
        "sub": str(user_id),
        "type": "access",
        "iat": now,
        "exp": now + (expires_delta or timedelta(hours=settings.security.jwt_expires_hours)),
    }
    return jwt.encode(payload, settings.security.jwt_secret, algorithm=settings.security.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """解码并校验JWT令牌，签名错误或已过期会抛出jwt.PyJWTError"""
    return jwt.decode(
        token,
        settings.security.jwt_secret,
        algorithms=[settings.security.jwt_algorithm],
        # 强制要求存在过期时间和用户标识，避免缺失声明的令牌被接受
        options={"require": ["exp", "sub"]},
    )
