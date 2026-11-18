from app.core.exceptions import ApplicationError


class UnauthorizedError(ApplicationError):
    """未登录或令牌无效/已过期"""

    status_code: int = 401
    code: str = "UNAUTHORIZED"
    message: str = "登录状态无效或已过期，请重新登录"


class InvalidCredentialsError(ApplicationError):
    """用户名或密码错误"""

    status_code: int = 401
    code: str = "INVALID_CREDENTIALS"
    message: str = "用户名或密码错误"


class UsernameAlreadyExistsError(ApplicationError):
    """用户名已被注册"""

    status_code: int = 409
    code: str = "USERNAME_ALREADY_EXISTS"
    message: str = "用户名已被注册"


class EmailAlreadyExistsError(ApplicationError):
    """邮箱已被注册"""

    status_code: int = 409
    code: str = "EMAIL_ALREADY_EXISTS"
    message: str = "邮箱已被注册"


class AccountDisabledError(ApplicationError):
    """账户已被禁用"""

    status_code: int = 403
    code: str = "ACCOUNT_DISABLED"
    message: str = "账户已被禁用，请联系客服"
