"""在线评测兼容垫片：必须先于 ragas 导入。

ragas 0.4.x 无条件导入 langchain_community.chat_models.vertexai，
而新版 langchain-community 已移除它；这里注册一个空壳顶替。
内容与 rag_eval/compat.py 一致——生产代码不依赖评测脚本目录，故单独复制一份。
"""

import sys
import types

import langchain_core.language_models as _lcm

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _mod = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI(_lcm.BaseChatModel):
        pass

    _mod.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _mod
