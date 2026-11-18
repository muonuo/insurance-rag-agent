from typing import Annotated

import jwt
from fastapi import Header

from app.core.security import decode_access_token

from .exceptions import UnauthorizedError


def get_current_user_id(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> int:
    """从Authorization: Bearer <token>请求头中解析当前登录用户ID"""

    # 1.请求头缺失或格式不对
    if not authorization or not authorization.lower().startswith("bearer "):
        raise UnauthorizedError()

    # 2.解码令牌，签名错误或已过期都视为未登录
    token = authorization[7:].strip()
    try:
        payload = decode_access_token(token)
    except jwt.PyJWTError as e:
        raise UnauthorizedError() from e

    # 3.校验令牌类型与用户标识
    if payload.get("type") != "access" or "sub" not in payload:
        raise UnauthorizedError()
    try:
        return int(payload["sub"])
    except (TypeError, ValueError) as e:
        raise UnauthorizedError() from e
