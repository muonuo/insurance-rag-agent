from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.graph.state import CompiledStateGraph

from app.core.config import settings
from app.core.logging import get_logger
from .memory import SessionMemoryMiddleware, session_memory
from .tools import (
    get_user_profile,
    query_candidate_products,
    query_user_plans,
    retrieve_insurance_clause,
    save_insurance_plan,
    save_user_profile,
)
from .schemas import InsuranceAgentContext

logger = get_logger(__name__)

SYSTEM_PROMPT = """
你是“安心保”的智能保险顾问。
- 你需要使用专业、准确、容易理解的语言回答用户的保险问题。
- 当信息不足时，应先向用户追问，形成用户画像。
- 开始推荐或用户提到“我之前说过”时，先调用get_user_profile读取长期画像（跨会话记忆），画像中已有的信息不要重复追问。
- 当用户透露年龄、家庭结构、保费预算、已有保单、健康状况或投保偏好时，及时调用save_user_profile更新长期画像。
- 给用户推荐保险方案时，必须先查询候选产品，根据用户画像，推荐合适的保险组合，再调用工具保存方案。
- 只有当用户主动提到之前保存过的方案（如“我之前存的方案”“上次推荐的”“我的方案”）时，才调用query_user_plans查询；用户没有提到时不要主动查询。
- 回答保险产品的条款问题时，必须先调用retrieve_insurance_clause查询对应产品的保险条款，再根据检索结果回答。
- 调用retrieve_insurance_clause时必须把用户所说的产品全名填入product_name，由工具解析产品ID；不要猜测或编造product_id。工具提示没有找到该产品时，如实告知用户并请其确认产品名称，不要改用猜测的ID再次检索。
- 你的回答必须是基于检索到的保险条款，不要编造保险条款中没有的信息。如果没有相关条款，如实告知用户。
- 回答中的每个事实陈述必须紧跟[source_id]的引用标记。例如：这款产品的等待期为180天。[ref-001]
- 回答要简短、精确，不要长篇大论。
"""


# agent在整个系统中，只初始化一次，
async def init_insurance_advisor() -> CompiledStateGraph:
    # 1.初始化模型
    model = init_chat_model(
        model=settings.llm.chat_model,
        api_key=settings.llm.api_key,
        extra_body={"thinking": {"type": "disabled"}}
    )

    # 2.创建Agent，绑定model、tools、Checkpointer
    agent = create_agent(
        model=model,
        tools=[
            get_user_profile,
            save_user_profile,
            query_candidate_products,
            query_user_plans,
            retrieve_insurance_clause,
            save_insurance_plan,
        ],
        checkpointer=True,
        system_prompt=SYSTEM_PROMPT,
        context_schema=InsuranceAgentContext,
        # 会话短期记忆：发给模型的历史替换为「摘要 + 最近N条」，checkpointer仍存全量
        middleware=[SessionMemoryMiddleware(session_memory)],
    )

    logger.info("保险顾问Agent初始化完毕~✅️")
    return agent
