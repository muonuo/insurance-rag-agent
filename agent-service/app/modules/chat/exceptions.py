from app.core.exceptions import ApplicationError


class FeedbackTargetNotFound(ApplicationError):
    """目标消息不存在或不支持反馈（较早的消息/该轮未产生trace）"""

    status_code: int = 404
    code: str = "FEEDBACK_TARGET_NOT_FOUND"
    message: str = "该消息不支持反馈"


class FeedbackUnavailable(ApplicationError):
    """反馈功能未启用（Langfuse未配置）或上传失败"""

    status_code: int = 503
    code: str = "FEEDBACK_UNAVAILABLE"
    message: str = "反馈功能未启用或暂时不可用"


class MemoryUnavailable(ApplicationError):
    """长期记忆存储未初始化，无法清除画像"""

    status_code: int = 503
    code: str = "MEMORY_UNAVAILABLE"
    message: str = "长期记忆存储暂不可用"
