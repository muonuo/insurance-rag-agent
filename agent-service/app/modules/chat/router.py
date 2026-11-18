from typing import Annotated

from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from app.infra.database import get_session
from app.modules.auth.dependencies import get_current_user_id
from .schema import ChatFeedbackRequest, ChatRequest
from .service import ChatService

router = APIRouter(prefix="/api/v1/chat", tags=["与AI对话接口"])


def get_chat_service(
    request: Request,
    session: AsyncSession = Depends(get_session)
):
    # 1.创建ChatService
    agent = request.app.state.agent
    return ChatService(session, agent)

# sse两步走：
# 1.响应类型response_class=EventSourceResponse，标记响应格式是event_stream格式
# 2.返回值必须用ServerSentEvent包装

@router.post("", response_class=EventSourceResponse)
async def chat(
    request: ChatRequest,
    user_id: Annotated[int, Depends(get_current_user_id)],
    service: ChatService=Depends(get_chat_service)
):
    # Generator, 用yield不断生成内容，形成动态列表
    async for sse in service.chat_stream(user_id, request):
        yield sse


@router.delete("/profile", status_code=204)
async def clear_profile(
    user_id: Annotated[int, Depends(get_current_user_id)],
    service: ChatService = Depends(get_chat_service)
):
    """清除当前用户的长期记忆（画像）；会话历史不受影响"""

    await service.clear_profile(user_id)


@router.post("/feedback", status_code=204)
async def feedback(
    request: ChatFeedbackRequest,
    user_id: Annotated[int, Depends(get_current_user_id)],
    service: ChatService=Depends(get_chat_service)
):
    """用户对AI回复点赞/点踩，写入Langfuse挂在对应trace上"""

    await service.submit_feedback(user_id, request)

