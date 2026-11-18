"""环境兼容垫片：ragas 0.4.x 无条件导入 langchain_community 的 vertexai 聊天模块，
而新版 langchain-community 已移除它。在所有 import ragas 之前注册一个空壳顶替。

约定：rag_eval 内任何模块都先 import rag_eval.compat，再 import ragas。
"""

import sys
import types

import langchain_core.language_models as _lcm

_mod = types.ModuleType("langchain_community.chat_models.vertexai")


class ChatVertexAI(_lcm.BaseChatModel):
    pass


_mod.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _mod
