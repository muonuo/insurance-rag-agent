from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Command

from .profile import get_profile
from .schemas import InsuranceAgentContext, InsuranceAgentState, Intent
from .intent_router import build_route_context, router
from .chitchat import init_chitchat_agent
from .claim import init_claim_agent
from .insurance_advisor import init_insurance_advisor
from .safety import safety_node

# 1.定义节点

# 1.1.意图识别节点（注入Runtime取user_id、Store取长期画像，用于老用户的个性化问候）
async def route_node(
    state: InsuranceAgentState,
    runtime: Runtime[InsuranceAgentContext],
    store: BaseStore,
) -> Command[Intent | Literal["safety"]]:
    # 获取上一轮的路由结果
    previous_workflow = state.get("active_workflow", "")
    # 获取用户本轮的提问问题
    messages = state['messages']
    # 准备上下文：上一轮工作流 + 最近几轮对话摘要，多轮追问（"那个产品""继续"）也能正确路由
    context = build_route_context(messages, previous_workflow)
    # 读取用户长期画像：老用户打招呼时按画像生成个性化问候（无Store或无画像返回None，走固定话术）
    profile = await get_profile(store, runtime.context.user_id) if store is not None else None

    # 调用路由工具，完成意图识别
    result = await router.route(messages[-1].text, context, profile)

    if isinstance(result, str):
        # 当前识别出的是寒暄，直接返回固定回复结果
        return Command(
            update={
                "active_workflow": "chitchat",
                "previous_workflow": previous_workflow,
                "messages": [AIMessage(content=result)]
            },
            goto="safety"
        )

    return Command(
        update={
            "active_workflow": result.intent,
            "previous_workflow": previous_workflow
        },
        goto=result.intent
    )

# 1.2.保险方案推荐咨询节点
# def recommendation_node(state: InsuranceAgentState):
#     return {"messages": [AIMessage("保险方案推荐功能尚未完成~")]}

# 1.3.理赔咨询节点（已由理赔顾问Agent实现，见app/agents/claim.py）

# 1.4.闲聊节点
# def chitchat_node(state: InsuranceAgentState):
#     return {"messages": [AIMessage("哈哈，欢迎你来到安心保商城，今天聊点啥~")]}

# 1.5.人工
def human_handoff_node(state: InsuranceAgentState):
    return {"messages": [AIMessage("你的请求已转人工，请耐心等待~~")]}

# 1.6.fallback
def fallback_node(state: InsuranceAgentState):
    return {"messages": [AIMessage("无法识别您的意图，请再说一遍~~")]}

# 2.创建Graph，用Edge连接Node
async def init_insurance_agent(checkpointer, store: BaseStore):
    # 建造器（声明运行时Context Schema，供子Agent的工具读取user_id；
    # 编译时传入Store，会自动传播到所有子Agent的工具，用于跨会话长期记忆）
    builder = StateGraph(
        InsuranceAgentState,
        context_schema=InsuranceAgentContext,
    )

    # 添加节点
    builder.add_node("route", route_node)
    builder.add_node("recommendation_plan",await init_insurance_advisor())
    builder.add_node("chitchat", init_chitchat_agent())
    builder.add_node("claim", await init_claim_agent())
    builder.add_node("human_handoff", human_handoff_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("safety", safety_node)

    # 添加Edge，连Node：所有出口统一经过安全节点做合规检查；
    # safety用Command动态路由（中危改写后→END，高危→human_handoff再回safety收尾），
    # 因此safety不设静态出边
    builder.add_edge(START, "route")
    builder.add_edge("recommendation_plan", "safety")
    builder.add_edge("claim", "safety")
    builder.add_edge("human_handoff", "safety")
    builder.add_edge("fallback", "safety")
    builder.add_edge("chitchat", "safety")

    # 编译Graph（传入Store，向子图和工具传播长期记忆）
    return builder.compile(checkpointer=checkpointer, store=store)