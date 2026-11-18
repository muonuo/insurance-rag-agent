"""用户画像：基于LangGraph Store的跨会话长期记忆"""

from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

# 画像在Store中的键：命名空间按用户隔离，每个用户一份画像
PROFILE_KEY = "profile"


def profile_namespace(user_id: int) -> tuple[str, str]:
    """画像的Store命名空间：(profiles, 用户ID)"""
    return ("profiles", str(user_id))


class UserProfile(BaseModel):
    """用户长期画像：全部字段可选，随对话逐步充实"""

    age: int | None = Field(None, description="年龄或年龄段")
    family: str | None = Field(None, description="家庭结构，如：已婚，有一个3岁女儿")
    budget: str | None = Field(None, description="保费预算，如：每年8000元以内")
    existing_coverage: list[str] | None = Field(
        None, description="已购买的保险或已有保障，传空列表表示清空"
    )
    health_notes: str | None = Field(None, description="影响投保的健康告知信息")
    preferences: str | None = Field(None, description="投保偏好，如：偏好消费型、关注性价比")
    notes: str | None = Field(None, description="其他值得跨会话记住的信息")


def merge_profile(current: UserProfile | None, update: UserProfile) -> UserProfile:
    """不可变合并：用update中的非None字段覆盖current，返回新画像对象。

    None表示"本次没有该信息"，不覆盖已有值；existing_coverage传[]可显式清空。
    """
    base = current or UserProfile()
    changed = {
        field: value
        for field, value in update.model_dump().items()
        if value is not None
    }
    return base.model_copy(update=changed)


async def get_profile(store: BaseStore, user_id: int) -> UserProfile | None:
    """读取用户画像，尚未建立画像时返回None"""

    item = await store.aget(profile_namespace(user_id), PROFILE_KEY)
    if item is None:
        return None
    return UserProfile.model_validate(item.value)


async def save_profile(store: BaseStore, user_id: int, update: UserProfile) -> UserProfile:
    """读取-合并-写入用户画像，返回合并后的最新画像"""

    merged = merge_profile(await get_profile(store, user_id), update)
    await store.aput(profile_namespace(user_id), PROFILE_KEY, merged.model_dump())
    return merged


async def delete_profile(store: BaseStore, user_id: int) -> bool:
    """彻底删除用户画像（用户主动清除记忆）；返回是否确实删掉了内容"""

    item = await store.aget(profile_namespace(user_id), PROFILE_KEY)
    if item is None:
        return False
    await store.adelete(profile_namespace(user_id), PROFILE_KEY)
    return True


# 摘要时各字段的中文标签，按对问候有价值程度排序
FIELD_LABELS = {
    "age": "年龄",
    "family": "家庭",
    "budget": "预算",
    "existing_coverage": "已有保障",
    "health_notes": "健康备注",
    "preferences": "偏好",
    "notes": "备注",
}


def summarize_profile(profile: UserProfile | None) -> str | None:
    """把画像压成一行中文摘要供提示词使用；无画像或空画像返回None"""

    if profile is None:
        return None
    parts = []
    for field, label in FIELD_LABELS.items():
        value = getattr(profile, field)
        if not value:
            continue
        text = "、".join(value) if isinstance(value, list) else str(value)
        parts.append(f"{label}：{text}")
    return "；".join(parts) if parts else None
