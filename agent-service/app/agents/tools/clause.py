"""条款检索工具：RAG链路对LLM的唯一出口

产品ID由工具内部按产品名解析，不交给模型猜测——模型猜错ID会检索到
别的产品条款，答案与问题不匹配（离线评测中也表现为上下文为空/分数为0）。
"""

import json

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer

from app.agents.schemas import InsuranceAgentContext
from app.rag.capture import record_contexts
from app.rag.models import ParentChunk


def _last_user_text(runtime: ToolRuntime | None) -> str | None:
    """取本轮用户原话，用于缓存定位（模型生成的query措辞会浮动，原话更稳定）"""

    state = getattr(runtime, "state", None) or {}
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) and message.text:
            return message.text
    return None


async def _resolve_product_id(product_name: str) -> int | None:
    """按产品名解析上架产品的ID；未找到返回None"""

    from app.infra.database import AsyncSessionFactory
    from app.modules.product.repository import ProductRepository

    async with AsyncSessionFactory() as session:
        product = await ProductRepository(session).find_by_name(product_name)
        return product.id if product else None


async def _search_clauses(
    query: str,
    k: int,
    product_id: int,
    key_query: str | None = None,
) -> dict[str, dict] | None:
    """检索条款并组装编号片段；没有命中返回None"""

    from app.rag import retriever

    chunks: list[ParentChunk] = await retriever.retrieve(
        query, k, product_id, key_query=key_query
    )
    if not chunks:
        return None

    return {
        f"ref-{i:03d}": {
            "product_id": chunk.product_id,
            "clause_name": chunk.clause_name,
            "section_path": chunk.section_path,
            "content": chunk.content,
        }
        for i, chunk in enumerate(chunks, start=1)
    }


@tool
async def retrieve_insurance_clause(
    query: str,
    product_name: str,
    k: int = 5,
    product_id: int | None = None,
    runtime: ToolRuntime[InsuranceAgentContext] | None = None,
) -> str:
    """查询指定保险产品的条款内容。

    Args:
        query: 需要从保险条款中查询的问题。
        product_name: 保险产品名称，填写用户提到的产品全名（工具会自动解析产品ID）。
        k: 返回的文档的数量，默认5。
        product_id: 仅当已经通过其他工具拿到确切产品ID时填写，通常不需要。
    """

    # 1.解析产品ID：优先按名称解析，避免模型猜ID检索到别的产品条款
    resolved_id: int | None = None
    if product_name.strip():
        resolved_id = await _resolve_product_id(product_name)
        if resolved_id is None:
            return (
                f"没有找到名为「{product_name}」的保险产品，"
                "请先确认用户询问的产品名称，不要凭猜测的ID检索。"
            )
    elif product_id is not None:
        resolved_id = product_id

    if resolved_id is None:
        return "请提供要查询条款的保险产品名称。"

    # 2.检索并组装编号片段 {"ref-001": {"content": ""}}
    # 缓存定位优先用用户原话：同一问题重复问时，模型生成的query可能措辞不同
    sources = await _search_clauses(
        query, k, resolved_id, key_query=_last_user_text(runtime)
    )
    if sources is None:
        return "没有查询到相关保险条款。"

    # 3.记录评测用的检索上下文（未开启捕获时为空操作）
    record_contexts([item["content"] for item in sources.values()])

    # 4.把引用片段写入自定义事件流，发给前端（不进入LLM的消息历史）
    writer = get_stream_writer()
    writer({"type": "additional_info", "data": sources})

    # 5.以JSON格式返回给LLM
    return json.dumps(sources, ensure_ascii=False)
