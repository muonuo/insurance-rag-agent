import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi.sse import ServerSentEvent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.stream import CustomTransformer
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.memory import session_memory
from app.agents.schemas import AdditionalInfo, InsuranceAgentContext
from app.core.logging import get_logger
from app.core.observability import (
    create_langfuse_handler,
    get_langfuse_client,
    observe_config,
)
from app.evaluation.online_eval import maybe_schedule_eval
from app.modules.chat_thread.exceptions import ChatThreadNotFound
from app.modules.chat_thread.repository import ChatThreadRepository
from app.rag.capture import capture_contexts
from app.agents.profile import delete_profile
from .exceptions import FeedbackTargetNotFound, FeedbackUnavailable, MemoryUnavailable
from .schema import ChatFeedbackRequest, ChatRequest

logger = get_logger(__name__)

# 只有内容型工作流的回复才开放用户反馈；固定话术工作流不展示反馈按钮
FEEDBACK_ELIGIBLE_WORKFLOWS = {"recommendation_plan", "claim", "chitchat"}


class _AnswerBuffer:
    """按AI消息id累积本轮最终回答

    兼容两类事件：message-start后跟流式增量（delta），以及直接下发完整AIMessage。
    完整消息与同id增量二选一——完整消息覆盖增量，避免同一段文本重复拼接；
    最终取final_message_id对应的文本，避免把中间步骤（工具调用等）的输出算进回答。
    """

    def __init__(self) -> None:
        self._by_id: dict[str, list[str]] = {}
        self._orphan: list[str] = []
        self._current_id: str | None = None
        self._last_whole: str = ""

    def add_whole(self, message: AIMessage) -> None:
        text = message.text or ""
        if not text:
            return
        key = str(message.id or "")
        if key:
            self._by_id[key] = [text]
        else:
            self._orphan.append(text)
        self._last_whole = text

    def start(self, message_id: str) -> None:
        self._current_id = message_id
        self._by_id.setdefault(message_id, [])

    def add_delta(self, text: str) -> None:
        if not text:
            return
        if self._current_id is None:
            self._orphan.append(text)
        else:
            self._by_id.setdefault(self._current_id, []).append(text)

    def finish(self) -> None:
        self._current_id = None

    def final_text(self, final_message_id: str | None) -> str:
        if final_message_id:
            parts = self._by_id.get(final_message_id)
            if parts:
                return "".join(parts)
        if self._last_whole:
            return self._last_whole
        if self._orphan:
            return "".join(self._orphan)
        for parts in reversed(list(self._by_id.values())):
            if parts:
                return "".join(parts)
        return ""


class ChatService:

    def __init__(self, session: AsyncSession, agent: CompiledStateGraph):
        self.agent = agent
        self.repository = ChatThreadRepository(session)

    async def chat_stream(
        self,
        user_id: int,
        request: ChatRequest,
    ) -> AsyncIterator[ServerSentEvent]:
        """实现Agent流式对话的函数"""
        turn_started = time.perf_counter()

        # 1.校验，判断当前thread_id是否属于当前用户
        chat_thread = await self.repository.required_owner(user_id, request.thread_id)
        if not chat_thread:
            # 指定会话不属于当前用户
            raise ChatThreadNotFound()

        # 2.组织输入：有decision说明是对中断的确认，用Command恢复执行
        if request.decision:
            _input = Command(resume=request.decision)
        else:
            _input = {"messages": [HumanMessage(request.message)]}

        # 3.组织config和上下文
        # handler按轮创建：last_trace_id会被后续run覆盖，全局单例在并发对话下会串线
        langfuse_handler: Any | None = create_langfuse_handler()
        config = {
            "configurable": {"thread_id": request.thread_id},
            **observe_config(user_id, request.thread_id, handler=langfuse_handler),
        }
        context = InsuranceAgentContext(
            user_id=user_id, thread_id=str(request.thread_id)
        )

        # 4.记录最终AIMessage的id和本轮产生的引用信息
        final_message_id: str | None = None
        additional_info: AdditionalInfo = {}
        # 可观测性：回复字数、token用量（来自message-finish事件）
        answer_chars = 0
        usage_total = {"input_tokens": 0, "output_tokens": 0}
        answer_buffer = _AnswerBuffer()
        turn_trace_id: str | None = None

        # 5.捕获检索上下文必须包住事件流的创建和消费，否则工具里record_contexts是空操作
        with capture_contexts() as captured_contexts:
            # 调用Agent（原始事件循环，可同时捕获根Graph和子Agent的事件）
            stream = await self.agent.astream_events(
                _input,
                config,
                context=context,
                version="v3",
                transformers=[CustomTransformer],
            )

            async for event in stream:
                method = event["method"]
                # AI回复的文本增量
                if method == "messages":
                    data = event["params"]["data"][0]
                    if isinstance(data, AIMessage):
                        answer_buffer.add_whole(data)
                        answer_chars += len(data.text or "")
                        # 完整消息（寒暄固定回复/个性化问候）没有message-start事件，
                        # 从消息本身取id，保证反馈与trace映射可用
                        if final_message_id is None and data.id:
                            final_message_id = data.id
                        yield ServerSentEvent(data=data.text, event="message")
                    elif isinstance(data, dict):
                        delta = data.get("delta")
                        if delta and delta.get("text"):
                            text = delta["text"]
                            answer_buffer.add_delta(text)
                            answer_chars += len(text)
                            yield ServerSentEvent(data=text, event="message")
                        elif data.get("event") == "message-start" and data.get("id"):
                            # 第一个消息片段携带AIMessage的id
                            final_message_id = data.get("id")
                            answer_buffer.start(final_message_id)
                        elif data.get("event") == "message-finish":
                            answer_buffer.finish()
                            # 消息结束事件携带token用量，逐条累加（一轮可能含多次LLM调用）
                            usage = data.get("usage") or {}
                            usage_total["input_tokens"] += usage.get("input_tokens") or 0
                            usage_total["output_tokens"] += usage.get("output_tokens") or 0

                # 工具中写入的引用片段等附加信息
                if method == "custom":
                    data = event["params"]["data"]
                    if isinstance(data, dict) and data.get("type") == "additional_info":
                        additional_info = data.get("data", {})
                        yield ServerSentEvent(data=additional_info, event="additional_info")

                # 只取第一个非空trace_id：根run先出现，避免被子图run覆盖
                if turn_trace_id is None and langfuse_handler is not None:
                    turn_trace_id = getattr(langfuse_handler, "last_trace_id", None)

            # 流结束兜底（例如handler在首个事件之后才写入trace_id）
            if turn_trace_id is None and langfuse_handler is not None:
                turn_trace_id = getattr(langfuse_handler, "last_trace_id", None)

            # 6.把中断信息（如方案确认）发给前端
            interrupts = await stream.interrupts()
            for item in interrupts:
                # Interrupt对象取值；兼容直接是dict的情况
                yield ServerSentEvent(
                    data=getattr(item, "value", item),
                    event="interrupt",
                )

        # with退出后列表仍可用（capture_contexts只重置ContextVar，不清理bucket）
        contexts = list(dict.fromkeys(captured_contexts))

        # 7.把引用信息与trace映射写入State：引用供历史恢复，trace映射供用户反馈定位
        # 注意：中断期间调用aupdate_state会清除待处理的interrupt，
        # 导致刷新后确认卡片丢失，因此有中断时跳过写入，优先保中断。
        state_update: dict = {}
        if final_message_id and turn_trace_id:
            state_update["trace_ids"] = {final_message_id: turn_trace_id}
        if final_message_id and additional_info:
            state_update["additional_info"] = {final_message_id: additional_info}
        if state_update and not interrupts:
            await self.agent.aupdate_state(config, state_update)

        # 8.每轮汇总日志：一眼看出本轮耗时、token开销和是否产生引用/中断
        logger.info(
            "对话轮完成",
            thread_id=str(request.thread_id),
            user_id=user_id,
            mode="decision" if request.decision else "message",
            duration_ms=round((time.perf_counter() - turn_started) * 1000, 1),
            answer_chars=answer_chars,
            refs=len(additional_info),
            interrupts=len(interrupts),
            contexts=len(contexts),
            state_written=bool(state_update and not interrupts),
            **usage_total,
        )

        answer_text = answer_buffer.final_text(final_message_id)

        # 9.在线RAG评测：只做守卫判断并投递后台任务，微秒级，不阻塞done事件
        try:
            maybe_schedule_eval(
                user_input=request.message,
                answer=answer_text,
                contexts=contexts,
                trace_id=turn_trace_id,
                user_id=user_id,
                thread_id=str(request.thread_id),
                has_interrupt=bool(interrupts),
                is_decision=bool(request.decision),
            )
        except Exception:
            # 双保险：调度函数自身异常也绝不影响对话
            logger.exception("在线评测调度失败，已忽略")

        # 9.1 会话短期记忆：把本轮问答追加进 Redis 窗口（有中断/无回答时跳过）
        if not interrupts and answer_text:
            user_text = request.message or "（用户确认方案）"
            try:
                await session_memory.append_turn(
                    str(request.thread_id), user_text, answer_text
                )
            except Exception:
                # 记忆缓存失败不影响对话
                logger.exception("会话记忆追加失败，已忽略")

        # 10.下发本轮AI消息id：前端反馈(👍/👎)时带上它来定位消息
        # 需同时满足：有消息id、有trace（有LLM调用）、且是内容型工作流。
        # 固定话术（寒暄固定回复/human_handoff/fallback）不开放反馈，
        # 不发meta让前端不渲染按钮，避免点击后404或无意义反馈。
        feedback_eligible = False
        if final_message_id and turn_trace_id:
            try:
                snapshot = await self.agent.aget_state(
                    {"configurable": {"thread_id": request.thread_id}}
                )
                active_workflow = snapshot.values.get("active_workflow")
                feedback_eligible = active_workflow in FEEDBACK_ELIGIBLE_WORKFLOWS
            except Exception:
                # 拿不到状态时保守处理：不下发meta
                logger.exception("获取active_workflow失败，本轮不开放反馈")
        if feedback_eligible:
            yield ServerSentEvent(
                data={"message_id": final_message_id}, event="meta"
            )

        # 11.返回结束标识
        yield ServerSentEvent(data="END", event="done")

    async def clear_profile(self, user_id: int) -> bool:
        """清除当前用户的长期记忆（画像），不影响任何会话历史"""

        store = getattr(self.agent, "store", None)
        if store is None:
            raise MemoryUnavailable()
        deleted = await delete_profile(store, user_id)
        logger.info("用户画像已清除", user_id=user_id, existed=deleted)
        return deleted

    async def submit_feedback(
        self,
        user_id: int,
        request: ChatFeedbackRequest,
        client: Any | None = None,
    ) -> None:
        """把用户对某条AI回复的👍/👎写成Langfuse分数，挂到该轮trace"""

        # 1.校验会话归属
        chat_thread = await self.repository.required_owner(user_id, request.thread_id)
        if not chat_thread:
            raise ChatThreadNotFound()

        # 2.从会话State取该消息对应的trace_id（没有映射的是本功能上线前的旧消息）
        snapshot = await self.agent.aget_state(
            {"configurable": {"thread_id": request.thread_id}}
        )
        trace_id = (snapshot.values.get("trace_ids") or {}).get(request.message_id)
        if not trace_id:
            raise FeedbackTargetNotFound()

        # 3.写分数（BOOLEAN：1=满意 0=不满意）并立即冲刷，看板即时可见
        if client is None:
            client = get_langfuse_client()
            if client is None:
                raise FeedbackUnavailable()
        try:
            client.create_score(
                name="user_feedback",
                value=1.0 if request.rating == "up" else 0.0,
                trace_id=trace_id,
                data_type="BOOLEAN",
                comment=request.comment,
            )
            client.flush()
        except Exception:
            logger.exception(
                "用户反馈上传失败",
                thread_id=str(request.thread_id),
                message_id=request.message_id,
            )
            raise FeedbackUnavailable()
