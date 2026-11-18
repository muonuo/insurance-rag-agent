"""理赔查询工具：保单与理赔进度"""

from langchain.tools import ToolRuntime, tool
from langgraph.config import get_stream_writer

from app.agents.schemas import InsuranceAgentContext
from app.infra.database import AsyncSessionFactory
from app.modules.claim.service import ClaimService


@tool
async def query_user_policies(
    runtime: ToolRuntime[InsuranceAgentContext],
) -> list[dict]:
    """查询当前用户名下的所有保单。当用户咨询保单、保障内容或需要确认可理赔的产品时使用。"""

    async with AsyncSessionFactory() as session:
        service = ClaimService(session)
        policies = await service.list_policies(runtime.context.user_id)

    if not policies:
        return []

    # 把保单信息写到自定义事件流，供前端展示保单卡片
    writer = get_stream_writer()
    writer({"type": "additional_info", "data": {"policies": policies}})

    return policies


@tool
async def query_user_claims(
    runtime: ToolRuntime[InsuranceAgentContext],
) -> list[dict]:
    """查询当前用户所有理赔案件的进度。当用户咨询理赔进度、理赔结果时使用。"""

    async with AsyncSessionFactory() as session:
        service = ClaimService(session)
        claims = await service.list_claims(runtime.context.user_id)

    if not claims:
        return []

    return claims
