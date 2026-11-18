"""app.core.security 纯工具函数测试：密码哈希与JWT令牌"""

from datetime import timedelta

import jwt
import pytest

from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


class TestPassword:
    def test_hash_and_verify_roundtrip(self):
        hashed = hash_password("123456")
        assert hashed != "123456"
        assert verify_password("123456", hashed) is True

    def test_wrong_password_fails(self):
        hashed = hash_password("123456")
        assert verify_password("wrong-password", hashed) is False

    def test_malformed_hash_returns_false(self):
        # 历史占位数据是畸形bcrypt格式，校验应返回False而不是抛异常
        assert verify_password("123456", "$2a$12$demo.password.hash.placeholder") is False


class TestAccessToken:
    def test_roundtrip_claims(self):
        token = create_access_token(user_id=42)
        payload = decode_access_token(token)
        assert payload["sub"] == "42"
        assert payload["type"] == "access"
        assert payload["exp"] > payload["iat"]

    def test_expired_token_raises(self):
        token = create_access_token(user_id=1, expires_delta=timedelta(seconds=-1))
        with pytest.raises(jwt.ExpiredSignatureError):
            decode_access_token(token)

    def test_garbage_token_raises(self):
        with pytest.raises(jwt.PyJWTError):
            decode_access_token("not-a-jwt-token")

    def test_wrong_signature_raises(self):
        token = jwt.encode({"sub": "1"}, "another-secret", algorithm="HS256")
        with pytest.raises(jwt.PyJWTError):
            decode_access_token(token)
