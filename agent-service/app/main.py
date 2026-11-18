import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.exceptions import ApplicationError
from app.core.logging import configure_logging, get_logger
from app.core.observability import flush_langfuse, warm_langfuse
from app.evaluation.online_eval import drain_online_evals, log_online_eval_config
from app.agents.orchestrator import init_insurance_agent
import uvicorn

# 初始化日志系统
configure_logging(settings.logging.level)

logger = get_logger(__name__)

# 在启动前执行
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"【{settings.app.name}】应用启动中...")
    from app.infra.database import check_database, close_database
    from app.infra.checkpointer import init_checkpointer, close_checkpointer
    from app.infra.store import init_store
    from app.rag import close_rag

    try:
        # 1.初始化数据库连接池
        await check_database()
        # 2.初始化Checkpointer
        checkpointer = await init_checkpointer()
        # 3.初始化长期记忆Store（共用Checkpointer连接池，必须在其之后）
        store = await init_store()
        # 4.初始化Agent,并将其保存在FastAPI的全局State中，一切进入FastAPI的请求都共享这个State
        app.state.agent = await init_insurance_agent(checkpointer, store)
        # 5.预热Langfuse追踪并打印在线评测配置（未启用时为空操作）
        warm_langfuse()
        log_online_eval_config()
        yield
    finally:
        logger.info("应用关闭中... ")
        # 0.先等在线评测任务把分数排入队列，再冲刷Langfuse（顺序不能反）
        await drain_online_evals(timeout=5.0)
        flush_langfuse()
        # 4.关闭rag
        close_rag()
        # 5.关闭数据库连接池
        await close_database()
        # 6.关闭Checkpointer连接池
        await close_checkpointer()


app = FastAPI(
    title=settings.app.name,
    debug=settings.app.debug,
    lifespan=lifespan
)

# CORS配置，解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", summary="健康检测接口")
async def health_check() -> dict[str, str]:
    logger.info("执行健康检查")
    return {"status": "ok"}

# HTTP请求可观测性：记录每个请求的方法、路径、状态码和耗时。
# 注意：SSE接口（如/chat）的耗时只统计到响应头返回，完整流式时长看服务层的"对话轮完成"日志。
@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    # 健康检查走debug级别，避免刷屏
    log = logger.debug if request.url.path == "/health" else logger.info
    log(
        "HTTP请求",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
    )
    return response

# 统一处理异常信息，发现是ChatThreadNotFound，返回对应的错误状态码和错误信息
@app.exception_handler(ApplicationError)
def handle_exception(request: Request, exc: ApplicationError):
    logger.error(f"处理异常, 请求路径: {request.url}, 异常信息: {exc}")
    return JSONResponse(
        status_code=exc.status_code,  # http状态码
        content={
            "code": exc.code,        # 错误码， error_code
            "message": exc.message     # 错误消息，error_message
        }
    )

from app.modules.product.router import router as product_router
from app.modules.chat_thread.router import router as chat_thread_router
from app.modules.chat.router import router as chat_router
from app.modules.auth.router import router as auth_router
app.include_router(chat_router)
app.include_router(chat_thread_router)
app.include_router(product_router)
app.include_router(auth_router)

if __name__ == "__main__":
    import sys
    uvicorn.run(
        "app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=False,
        loop= "asyncio:SelectorEventLoop" if sys.platform == "win32" else "auto",
    )