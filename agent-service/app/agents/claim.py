from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.graph.state import CompiledStateGraph

from app.core.config import settings
from app.core.logging import get_logger
from .memory import SessionMemoryMiddleware, session_memory
from .schemas import InsuranceAgentContext
from .tools import (
    get_user_profile,
    query_user_claims,
    query_user_policies,
    retrieve_insurance_clause,
)

logger = get_logger(__name__)

SYSTEM_PROMPT = """
你是“安心保”的理赔顾问，负责处理报案、理赔责任、理赔材料、理赔流程和理赔进度咨询。
- 使用专业、准确、容易理解的语言回答，回答要简短、精确。
- 查询用户保单或理赔进度时，必须调用query_user_policies或query_user_claims工具，不要编造保单或理赔信息。
- 需要了解用户背景（如健康告知备注、家庭情况）时，可调用get_user_profile读取长期画像作为辅助参考；但保单和理赔数据必须以query_user_policies和query_user_claims的查询结果为准，画像信息不得替代真实保单数据。
- 回答理赔责任、理赔材料、免赔额等条款问题时，必须先通过query_user_policies确认用户的保单和产品，再调用retrieve_insurance_clause查询对应产品条款，依据检索结果回答。
- 调用retrieve_insurance_clause时把产品名称填入product_name，由工具解析产品ID；不要猜测或编造product_id。工具提示没有找到该产品时，如实告知用户并请其确认产品名称。
- 回答中的每个事实陈述必须紧跟[source_id]的引用标记。例如：住院医疗的免赔额为1万元。[ref-001]
- 没有查询到相关保单或条款时，如实告知用户，不要编造。
- 涉及赔付金额争议、投诉或用户明确要求人工服务时，告知用户已为其转接人工客服。
"""


async def init_claim_agent() -> CompiledStateGraph:
    # 1.初始化模型
    model = init_chat_model(
        model=settings.llm.chat_model,
        api_key=settings.llm.api_key,
        extra_body={"thinking": {"type": "disabled"}},
    )

    # 2.创建Agent，绑定model、tools、Checkpointer和Context
    agent = create_agent(
        model=model,
        tools=[
            get_user_profile,
            query_user_policies,
            query_user_claims,
            retrieve_insurance_clause,
        ],
        checkpointer=True,
        system_prompt=SYSTEM_PROMPT,
        context_schema=InsuranceAgentContext,
        middleware=[SessionMemoryMiddleware(session_memory)],
    )

    logger.info("理赔顾问Agent初始化完毕~✅️")
    return agent
