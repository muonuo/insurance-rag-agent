import logging
from pathlib import Path

from pydantic import Field, computed_field, field_validator, model_validator, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

# 以当前文件为基准，找向上2级的父目录，也就是项目的根目录
APP_ROOT = Path(__file__).resolve().parents[2]

class EnvSettings(BaseSettings):
    """公共settings，配置.env文件位置"""
    model_config = SettingsConfigDict(
        env_file=(APP_ROOT / ".env",),
        env_file_encoding="utf-8",
        extra="ignore",  # settings中未定义的字段被忽略
    )

class AppSettings(EnvSettings):
    """应用相关设置"""
    name: str = Field(alias="APP_NAME")
    host: str = Field(alias="APP_HOST", default="127.0.0.1")
    port: int = Field(alias="APP_PORT", default=8001)
    debug: bool = Field(alias="APP_DEBUG", default=True, description="是否开启FastAPI的DEBUG模式")

class DatabaseSettings(EnvSettings):
    """数据库相关设置"""
    host: str = Field(alias="DB_HOST", default="localhost")
    port: int = Field(alias="DB_PORT", default=5432)
    name: str = Field(alias="DB_NAME", default="insurance")
    user: str = Field(alias="DB_USER", default="insurance")
    password: str = Field(alias="DB_PASSWORD", default="insurance123")
    echo: bool = Field(alias="DB_ECHO", default=False)
    pool_size: int = Field(alias="DB_POOL_SIZE", default=5)
    max_overflow: int = Field(alias="DB_MAX_OVERFLOW", default=10)
    pool_timeout: int = Field(alias="DB_POOL_TIMEOUT", default=30)
    pool_recycle: int = Field(alias="DB_POOL_RECYCLE", default=1800)

    def build_url(self, scheme: str):
        # 自动拼接URL路径
        return PostgresDsn.build(
            scheme=scheme,
            host=self.host,
            port=self.port,
            username=self.user,
            password=self.password
        ).encoded_string()

    @computed_field
    @property
    def url(self) -> str:
        # 自动拼接URL路径
        return self.build_url("postgresql+asyncpg")

    @computed_field
    @property
    def checkpoint_url(self) -> str:
        # 自动拼接URL路径
        return self.build_url("postgresql")

class LLMSettings(EnvSettings):
    """模型相关配置"""
    api_key: str = Field(alias="DEEPSEEK_API_KEY")
    chat_model: str = Field(alias="CHAT_MODEL", default="deepseek-flash")
    dashscope_api_key: str = Field(alias="DASHSCOPE_API_KEY")
    dashscope_base_url: str = Field(alias="DASHSCOPE_BASE_URL")

class LoggingSettings(EnvSettings):
    """日志相关配置"""
    level: str = Field(alias="LOGGING_LEVEL", default="INFO") # 可选值：DEBUG、INFO、WARNING、ERROR、CRITICAL

class RAGSettings(EnvSettings):
    """RAG相关配置"""
    mineru_token: str = Field(alias="MINERU_TOKEN")
    milvus_host: str = Field(alias="MILVUS_HOST", default="127.0.0.1")
    milvus_port: int = Field(alias="MILVUS_PORT", default=19530)
    # RRF融合平滑常数，越大越弱化头部名次优势，推荐10~100
    rrf_k: int = Field(alias="RRF_K", default=60)
    # 第二阶段精排使用的模型（DashScope）
    rerank_model: str = Field(alias="RERANK_MODEL", default="gte-rerank-v2")
    # 检索结果L1缓存（Redis）：同样的(query,k,product_id)直接返回，跳过混合检索+重排+回表
    cache_enabled: bool = Field(alias="RAG_CACHE_ENABLED", default=True)
    cache_redis_url: str = Field(alias="RAG_CACHE_REDIS_URL", default="redis://localhost:6380/0")
    # 缓存过期秒数（条款重新入库时会主动失效，TTL只是兜底）
    cache_ttl: int = Field(alias="RAG_CACHE_TTL", default=3600)

    @computed_field
    @property
    def milvus_url(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"

class ObservabilitySettings(EnvSettings):
    """Langfuse可观测性配置。enabled默认关闭：未配置key时追踪零开销"""

    enabled: bool = Field(alias="LANGFUSE_ENABLED", default=False)
    public_key: str = Field(alias="LANGFUSE_PUBLIC_KEY", default="")
    secret_key: str = Field(alias="LANGFUSE_SECRET_KEY", default="")
    # 自托管看板地址，见deploy/langfuse-compose.yml
    host: str = Field(alias="LANGFUSE_HOST", default="http://localhost:3100")

class OnlineEvalSettings(EnvSettings):
    """在线RAG评测配置。需同时启用Langfuse才能回写分数"""

    enabled: bool = Field(alias="ONLINE_EVAL_ENABLED", default=False)
    sample_rate: float = Field(
        alias="ONLINE_EVAL_SAMPLE_RATE", default=0.1, ge=0.0, le=1.0
    )
    # answer_relevancy 需要embedding；默认与离线评测一致指向本地Ollama
    embedding_base_url: str = Field(
        alias="ONLINE_EVAL_EMBEDDING_BASE_URL", default="http://localhost:11434/v1"
    )
    embedding_model: str = Field(
        alias="ONLINE_EVAL_EMBEDDING_MODEL", default="qwen3-embedding:0.6b"
    )


class SessionMemorySettings(EnvSettings):
    """会话短期记忆：Redis缓存最近消息+更早历史摘要，压缩发给模型的历史"""

    enabled: bool = Field(alias="SESSION_MEMORY_ENABLED", default=True)
    redis_url: str = Field(
        alias="SESSION_MEMORY_REDIS_URL", default="redis://localhost:6380/1"
    )
    ttl: int = Field(alias="SESSION_MEMORY_TTL", default=3600)
    # 保留原文的最近消息条数，更早的进入摘要
    recent_messages: int = Field(alias="SESSION_MEMORY_RECENT_MESSAGES", default=10)


class SecuritySettings(EnvSettings):
    """安全认证相关设置"""
    # 开发环境默认密钥，生产环境必须在.env中配置JWT_SECRET
    jwt_secret: str = Field(
        alias="JWT_SECRET",
        default="dev-insecure-secret-change-me",
    )
    jwt_algorithm: str = Field(alias="JWT_ALGORITHM", default="HS256")
    # 令牌有效期（小时），默认7天
    jwt_expires_hours: int = Field(alias="JWT_EXPIRES_HOURS", default=168)

    @field_validator("jwt_secret")
    @classmethod
    def warn_default_secret(cls, value: str) -> str:
        if value == "dev-insecure-secret-change-me":
            # 此处用标准库logging，避免与app.core.logging产生循环导入
            logging.getLogger(__name__).warning(
                "JWT_SECRET未配置，正在使用开发默认值，切勿用于生产环境！"
            )
        return value

class Settings(BaseSettings):
    app: AppSettings = Field(default_factory=AppSettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    online_eval: OnlineEvalSettings = Field(default_factory=OnlineEvalSettings)
    session_memory: SessionMemorySettings = Field(default_factory=SessionMemorySettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)

    @model_validator(mode="after")
    def require_production_secret(self) -> "Settings":
        """非调试环境必须配置真实的JWT密钥，避免使用公开默认值导致令牌可被伪造"""
        if self.security.jwt_secret == "dev-insecure-secret-change-me" and not self.app.debug:
            raise ValueError("非调试环境必须配置 JWT_SECRET，不能使用默认开发密钥")
        return self

# 项目启动直接初始化
settings = Settings()