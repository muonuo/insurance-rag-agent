from langgraph.store.base import BaseStore
from langgraph.store.postgres import AsyncPostgresStore

from app.core.logging import get_logger
from app.infra.checkpointer import checkpoint_pool

logger = get_logger(__name__)


# Store与Checkpointer共用一个连接池：两者对连接的要求完全一致
# （autocommit=True、prepare_threshold=0、dict_row），分池反而浪费连接。
# 注意：因此本模块必须在init_checkpointer之后调用（连接池在那里打开）。
async def init_store() -> BaseStore:
    """初始化长期记忆Store并完成建表迁移"""

    store = AsyncPostgresStore(checkpoint_pool)
    await store.setup()
    logger.info("长期记忆Store初始化成功~✅️")
    return store
