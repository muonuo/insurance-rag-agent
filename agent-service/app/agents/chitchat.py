from typing import NotRequired

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph

from app.core.config import settings
from app.core.logging import get_logger
from .schemas import InsuranceAgentState

logger = get_logger(__name__)

SYSTEM_MESSAGE = """
你是《安心保》商城的智能客服，负责处理与保险业务无关的普通聊天。
    
要求：
1. 回复友好、自然，最多两句话。
2. 不假装有个人经历、情感、实时信息或外部能力。
3. 不回答医疗、法律、投资、政治等高风险专业建议。
4. 每次回复都自然引导用户回到保险产品、投保或理赔业务。
5. 超出能力范围时，如实说明无法提供。
""".strip()


# 1.定义State
class ChitchatState(InsuranceAgentState):
    """闲聊子Agent状态：共享主Graph的消息与工作流字段，只额外维护闲聊次数"""

    chitchat_count: NotRequired[int]


# 2.定义Workflow

MAX_CHITCHAT_COUNT = 3   # 最大闲聊轮次


class ChitChatWorkflow:

    def __init__(self):
        self.model = init_chat_model(
            model=settings.llm.chat_model,
            api_key=settings.llm.api_key,
            extra_body={"thinking": {"type": "disabled"}},
        )

    async def chitchat(self, state: ChitchatState) -> dict:
        """处理闲聊：超过最大轮次返回固定回复，否则由模型自然回复"""

        # 取出State中的闲聊次数
        count = state.get("chitchat_count", 0)

        # 上一轮不是闲聊，说明进入新一轮闲聊，重置计数
        if state.get("previous_workflow") != "chitchat":
            count = 0

        # 超过最大轮次，返回固定回复，不再消耗模型资源
        if count >= MAX_CHITCHAT_COUNT:
            logger.info("闲聊次数超过最大值，直接返回固定结果。")
            return {
                "messages": [
                    AIMessage("我主要协助处理保险产品、投保和理赔咨询，请告诉我想办理或了解的事项。")
                ]
            }

        logger.info(f"当前闲聊次：{count}")
        # 异步调用模型，避免阻塞事件循环
        response = await self.model.ainvoke(
            [SystemMessage(SYSTEM_MESSAGE), *state["messages"]]
        )

        return {
            "messages": [response],
            "chitchat_count": count + 1,
        }

    def build_chitchat_graph(self):
        # 创建Builder
        builder = StateGraph(ChitchatState)

        # 添加节点
        builder.add_node("chitchat", self.chitchat)

        # 添加边
        builder.set_entry_point("chitchat")     # START -> chitchat
        builder.set_finish_point("chitchat")    # chitchat -> END

        # 编译，使用父Graph提供的Checkpointer保存子图私有状态
        return builder.compile(checkpointer=True)


def init_chitchat_agent():
    return ChitChatWorkflow().build_chitchat_graph()
