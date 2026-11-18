from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    """基础对话请求：既可以是新消息，也可以是对上一次中断的确认结果"""

    thread_id: UUID
    message: str | None = Field(default=None, min_length=1)
    decision: dict[str, Any] | None = None

    @model_validator(mode="after")
    def check_message_or_decision(self):
        """必须提供新消息或中断确认结果其中之一"""
        if not self.message and not self.decision:
            raise ValueError("message 和 decision 至少提供一个")
        return self


class ChatFeedbackRequest(BaseModel):
    """用户对某条AI回复的反馈（👍/👎）"""

    thread_id: UUID
    message_id: str = Field(min_length=1)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=500)
