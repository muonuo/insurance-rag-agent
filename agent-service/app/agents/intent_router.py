import re
import time
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.core.config import settings
from app.core.logging import get_logger
from .profile import UserProfile, summarize_profile
from .schemas import RouteResult
logger = get_logger(__name__)


SYSTEM_PROMPT = """
你是保险商城智能客服的意图识别节点，请判断用户消息应该交给哪个工作流。

intent只能从以下选项中选择：
- recommendation_plan：保险产品咨询、产品条款、保险推荐、方案管理和投保咨询
- claim：报案、理赔责任、理赔材料、理赔流程和理赔进度
- human_handoff：投诉、高风险问题或用户明确要求人工服务
- chitchat：与保险业务无关的普通聊天
- fallback：无法判断用户意图

历史上下文提供了上一轮工作流和最近几轮对话摘要；遇到"那个""继续""再详细点"等
指代性问题时，请结合上下文判断意图，而不是只看当前消息的字面。
""".strip()

GREETING_SYSTEM_PROMPT = """
你是"安心保"商城的保险顾问。老用户向你打招呼，请结合他的画像自然地回应：
1. 只能使用给定画像中的事实。严禁编造画像里没有的信息，尤其是：
   - 不要编造称呼或姓氏（不要叫"王先生/李女士"等，统一用"您好"）；
   - 不要编造"你之前说过…""上次提到…"这类历史，除非最近对话里确实出现过。
2. 先一句自然的问候，再根据画像中最值得跟进的一点主动问一个具体问题。
3. 最多两句话，口语化；一次只问一个问题，不要罗列信息、不要推销。
""".strip()

class IntentRouter:

    def __init__(self):
        # 初始化模型
        model = init_chat_model(
            model=settings.llm.chat_model,
            api_key=settings.llm.api_key,
            extra_body={"thinking": {"type": "disabled"}},
        )
        # 给模型绑定结构化输出，让模型按照要求输出路由结果
        self.route_model = model.with_structured_output(RouteResult).with_config({
            "tags": ["nostream"]
        })
        # 个性化问候用普通文本输出（老用户打招呼时按画像寒暄），同样不进前端事件流
        self.greet_model = model.with_config({"tags": ["nostream"]})

    async def route(
        self,
        query: str, # 用户本轮的提问
        context: str,    # 历史对话的上下文（上一轮工作流 + 最近几轮对话摘要）
        profile: UserProfile | None = None,   # 用户长期画像，有画像的问候会个性化
    ) -> RouteResult | str:
        """返回str表示寒暄回复（固定或个性化），返回RouteResult表示模型识别结果"""
        started = time.perf_counter()
        try:
            # 1.寒暄识别（基于规则，正则表达式匹配）
            intent = match_social_intent(query)
            if intent:
                # 打招呼且已有画像：按画像生成个性化问候，其余寒暄保持固定话术
                profile_summary = summarize_profile(profile)
                if intent == "greeting" and profile_summary:
                    return await self.personal_greeting(query, context, profile_summary)
                # 识别出是寒暄中的一种，返回寒暄固定回复
                return response_by_social_intent(intent)

            # 2.基于模型识别意图（异步调用，避免阻塞事件循环）
            result: RouteResult = await self.route_model.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"用户问题:{query}\n 历史上下文：{context}\n")
            ])
        except Exception:
            # 兜底方案，意图识别失败（如模型/网络故障），返回fallback
            logger.exception("模型意图识别失败")
            result = RouteResult(intent="fallback", reason="意图识别出现异常")

        logger.info(
            "意图识别完成",
            intent=result.intent,
            reason=result.reason,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return result

    async def personal_greeting(self, query: str, context: str, profile_summary: str) -> str:
        """按画像生成个性化问候；生成失败或空回复时退回固定话术，绝不阻塞寒暄"""

        started = time.perf_counter()
        try:
            response = await self.greet_model.ainvoke([
                SystemMessage(content=GREETING_SYSTEM_PROMPT),
                HumanMessage(content=f"用户画像：{profile_summary}\n\n最近对话：\n{context}\n\n用户说：{query}"),
            ])
            if response.text:
                logger.info(
                    "个性化问候",
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )
                return response.text
        except Exception:
            logger.exception("个性化问候生成失败，退回固定话术")
        return SOCIAL_RESPONSES["greeting"]


# 路由可见的最近历史消息条数与单条截断长度（控制token开销）
ROUTE_CONTEXT_MESSAGES = 6
ROUTE_CONTEXT_MAX_CHARS = 200


def build_route_context(messages: list[BaseMessage], active_workflow: str) -> str:
    """构造意图识别的历史上下文：上一轮工作流 + 最近几轮可展示对话（逐条截断）"""

    # 1.排除本轮用户消息（它作为query单独传给路由），只看历史
    history = messages[:-1]
    # 2.挑出可展示的消息：用户提问与AI正文回复，跳过工具调用等中间过程
    visible = [
        message
        for message in history
        if isinstance(message, (HumanMessage, AIMessage))
        and not getattr(message, "tool_calls", None)
        and message.text
    ][-ROUTE_CONTEXT_MESSAGES:]
    # 3.带角色标注拼装
    lines = [f"上一轮工作流：{active_workflow or '无'}"]
    for message in visible:
        role = "用户" if isinstance(message, HumanMessage) else "AI"
        lines.append(f"{role}: {message.text[:ROUTE_CONTEXT_MAX_CHARS]}")
    return "\n".join(lines)


# 基于规则的识别是否为寒暄

# 寒暄类型枚举
SocialIntent = Literal["greeting", "thanks", "goodbye"]
# 寒暄的匹配规则
SOCIAL_PATTERNS = {
    "greeting": (
        r"(你|您)?好(呀|啊|哦)?",
        r"嗨",
        r"哈喽",
        r"hello",
        r"hi",
        r"在吗",
    ),
    "thanks": (
        r"谢谢(你|您)?",
        r"多谢(你|您)?",
        r"感谢(你|您)?",
        r"辛苦了",
    ),
    "goodbye": (
        r"再见",
        r"拜拜",
        r"先这样",
        r"没事了",
    ),
}
# 寒暄的固定回复
SOCIAL_RESPONSES = {
    "greeting": "您好，我是您的保险顾问，可以帮您咨询产品、推荐方案或办理理赔。",
    "thanks": "不客气，后续有保险问题可以随时问我。",
    "goodbye": "好的，后续需要帮助时随时联系我。",
}

def normalize_message(message: str) -> str:
    """格式化消息，去掉无效字符"""

    return re.sub(r"[\s，。！？、,.!?~～]", "", message).lower()


def match_social_intent(message: str) -> SocialIntent | None:
    """基于规则匹配的路由"""

    normalized = normalize_message(message)
    for intent, patterns in SOCIAL_PATTERNS.items():
        if any(re.fullmatch(pattern, normalized) for pattern in patterns):
            return intent
    return None

def response_by_social_intent(intent: SocialIntent):
    """根据规则识别出的意图返回固定回复"""

    return SOCIAL_RESPONSES.get(intent)



# 创建好对象，方便对方使用
router = IntentRouter()