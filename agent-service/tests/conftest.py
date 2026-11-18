"""pytest环境准备：保证在没有本地.env的干净检出处也能运行测试"""

import os

# 必须在导入app模块之前设置：Settings中的这些字段是必填项
os.environ.setdefault("APP_NAME", "agent-service-tests")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-deepseek-key")
os.environ.setdefault("DASHSCOPE_API_KEY", "test-dashscope-key")
os.environ.setdefault("DASHSCOPE_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("MINERU_TOKEN", "test-mineru-token")
os.environ.setdefault("APP_DEBUG", "true")
# 测试专用密钥，避免依赖真实配置
os.environ.setdefault("JWT_SECRET", "unit-test-secret-do-not-use-in-production")
# 单测不连真实Redis：会话记忆默认关闭，需要时在用例里单独构造实例
os.environ.setdefault("SESSION_MEMORY_ENABLED", "false")
