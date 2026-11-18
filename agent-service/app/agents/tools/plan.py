"""方案工具：保存（interrupt人工确认）与查询用户已保存方案"""

import json

from fastapi.encoders import jsonable_encoder
from langchain.tools import ToolRuntime, tool
from langgraph.types import interrupt

from app.agents.schemas import InsuranceAgentContext
from app.infra.database import AsyncSessionFactory
from app.modules.insurance_plan.schemas import InsurancePlanCreate
from app.modules.insurance_plan.service import InsurancePlanService


async def _fetch_plans(user_id: int) -> list[dict]:
    """查询用户保存过的方案（独立函数便于测试替换）"""

    async with AsyncSessionFactory() as session:
        service = InsurancePlanService(session)
        return await service.list_plans(user_id)


@tool
async def query_user_plans(
    runtime: ToolRuntime[InsuranceAgentContext],
) -> str:
    """查询当前用户之前保存过的保险方案（跨会话记忆）。

    只在用户主动提到"之前保存的方案/上次推荐过的/我的方案"时调用；
    用户没有提到时不要调用，避免每轮都查。返回方案列表及其产品明细。
    """

    plans = await _fetch_plans(runtime.context.user_id)
    if not plans:
        return "用户还没有保存过保险方案。"

    return json.dumps(jsonable_encoder(plans), ensure_ascii=False)


@tool
async def save_insurance_plan(
    plan: InsurancePlanCreate,
    runtime: ToolRuntime[InsuranceAgentContext],
) -> dict[str, str]:
    """
    请用户确认后，保存当前用户的保险推荐方案。
    当你查询完候选产品后，根据用户画像、候选产品生成保险组合方案，交由用户确认并保存。
    方案中每个方案项的product_id必须使用候选产品查询结果中的真实产品ID。
    """

    # 中断执行，把方案发给用户确认
    decision = interrupt(
        {
            "plan": plan.model_dump(mode="json"),
            "confirm_info": "是否确认保存这份保险推荐方案？",
            "action": ["approve", "reject"],
        }
    )
    # 用户拒绝，则不再保存
    if decision.get("action") != "approve":
        return {"message": f"用户取消保存保险方案，原因：{decision.get('reject_reason')}"}

    async with AsyncSessionFactory() as session:
        service = InsurancePlanService(session)
        plan_id = await service.create_plan(
            user_id=runtime.context.user_id,
            data=plan,
        )

    return {
        "plan_id": str(plan_id),
        "message": "保险方案保存成功",
    }
