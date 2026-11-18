"""会话记忆中间件：只替换「发给模型的消息」，不改 state/checkpointer

每轮模型调用前，把"当前轮之前的历史"换成 Redis 里的紧凑视图：
  [更早对话摘要] + [最近N条原文] + [当前轮全部消息（含工具调用）]
当前轮的消息原样保留，保证 ReAct 工具循环正常；摘要/缓存不可用时退回窗口截断。
"""

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage

from .session_memory import SessionMemory, _dict_to_message, _is_visible, split_history_and_current


class SessionMemoryMiddleware(AgentMiddleware):
    def __init__(self, memory: SessionMemory):
        self.memory = memory

    async def awrap_model_call(self, request: ModelRequest, handler):
        context = getattr(getattr(request, "runtime", None), "context", None)
        thread_id = getattr(context, "thread_id", None)
        if not thread_id or not self.memory.enabled:
            return await handler(request)

        history, current = split_history_and_current(request.messages)
        snapshot = await self.memory.get(thread_id)

        if snapshot is not None and snapshot.recent:
            compact_history = [_dict_to_message(item) for item in snapshot.recent]
        else:
            # 缓存未命中：先用窗口截断兜底，并触发后台摘要，下一轮生效
            visible = [m for m in history if _is_visible(m)]
            compact_history = visible[-self.memory.recent_limit :]

        # 有新的溢出内容时后台刷新摘要（不阻塞本轮）
        self.memory.maybe_schedule_refresh(thread_id, history, snapshot)

        compact = []
        if snapshot is not None and snapshot.summary:
            compact.append(SystemMessage(content=f"更早对话摘要：\n{snapshot.summary}"))
        compact.extend(compact_history)
        compact.extend(current)

        return await handler(request.override(messages=compact))
