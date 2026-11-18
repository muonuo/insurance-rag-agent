"""评测配置：复用主项目 Settings（自动加载 .env），只补充评测专用参数。"""

import rag_eval.compat  # noqa: F401  必须先于 ragas 加载

from pathlib import Path

from app.core.config import settings

EVAL_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = EVAL_DIR / "resources"
RESULTS_DIR = EVAL_DIR / "results"

QUESTIONS_PATH = RESOURCES_DIR / "insurance_eval_dataset.json"
DATASET_NAME = "insurance_rag_eval"

# 评测使用演示用户；涉及保单/方案的工具依赖真实用户ID
EVAL_USER_ID = 1
# 每题间隔，避免触发模型限流
REQUEST_DELAY = 1.0

# ---- 判卷模型（与业务模型解耦，便于单独替换）----
# 文本裁判：DeepSeek
JUDGE_LLM = settings.llm.chat_model
DEEPSEEK_API_KEY = settings.llm.api_key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# 向量裁判：本地 Ollama（与主项目 embedding 模型保持一致）
JUDGE_EMBEDDING_BASE_URL = "http://localhost:11434/v1"
JUDGE_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
