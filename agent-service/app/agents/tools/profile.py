"""长期画像工具：读写跨会话用户画像（写入方仅保险顾问）"""

from langchain.tools import ToolRuntime, tool

from app.agents.profile import UserProfile, get_profile, save_profile
from app.agents.schemas import InsuranceAgentContext


@tool
async def get_user_profile(
    runtime: ToolRuntime[InsuranceAgentContext],
) -> str:
    """获取当前用户的长期画像（跨会话记忆）。
    开始推荐保险方案、或用户提到"我之前告诉过你"时，先调用此工具了解用户情况。
    """

    profile = await get_profile(runtime.store, runtime.context.user_id)
    if profile is None:
        return "暂无用户画像，请通过对话了解用户后再推荐。"
    return profile.model_dump_json()


@tool
async def save_user_profile(
    profile: UserProfile,
    runtime: ToolRuntime[InsuranceAgentContext],
) -> dict:
    """当用户透露年龄、家庭结构、保费预算、已有保单、健康状况或投保偏好时，
    调用此工具更新用户长期画像（跨会话记忆）。只需传入本次新增或变化的字段，已有字段保持不变。
    """

    merged = await save_profile(runtime.store, runtime.context.user_id, profile)
    return {"message": "用户画像已更新", "profile": merged.model_dump()}
