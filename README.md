# 安心保 · 保险智能顾问 Agent

面向投保场景的 **RAG + 多 Agent** 智能顾问：条款问答、方案推荐、理赔指引，一套工程化的 Agent 后端。

后端 FastAPI + SSE 流式输出，前端 React + Ant Design，依赖全部通过 Docker Compose 自托管（PostgreSQL / Milvus / Redis / Langfuse）。

知识库来自 **10 份真实保险条款 + 8 份费率表**，覆盖重疾险、医疗险、意外险、定期寿险四大类共 **12 款在售产品**。

---

## 功能特性

### 多 Agent 协作
- 主控路由 + 三个专业子 Agent（保险顾问 / 理赔助手 / 闲聊）自动分派
- 路由用 `Command` 动态跳转，支持多轮追问的指代消解（"那个""再详细点"）
- 子 Agent 通过 Function Calling 调用各自工具集；闲聊超过 3 轮自动引导回业务，减少无效调用

### 记忆机制
- **长期记忆**：Checkpointer 持久化会话原文；Store 按 `user_id` 维护跨会话画像（不可变合并），自动沉淀年龄、家庭、预算等信息
- **短期记忆**：Redis 缓存「最近 10 条原文 + 更早历史滚动摘要」，增量更新、TTL 兜底；缓存未命中或 Redis 故障时自动降级为窗口截断
- 跨会话方案查询工具、方案入库前 `interrupt` 人工确认

### RAG 全链路
- MinerU 解析 PDF + LLM 修复表格错乱，父子块切分（子块检索、父块回填）
- BM25 + 向量混合召回 → RRF 融合 → Cross-Encoder 两阶段重排，**限定在目标产品内**
- 产品名由服务端解析为 `product_id`，避免模型猜测导致检索到错误产品
- 回答强制附带 `[ref-xxx]` 引用溯源

### 检索缓存
- Redis L1 精确匹配缓存：key 按归一化查询 + 产品维度生成，优先用用户原话定位
- 条款重新入库时按产品批量失效，TTL 兜底，故障时自动降级并统计命中率

### 合规审核
- 所有回答出口统一经过审核节点：PII 自动脱敏、违规销售话术改写
- 诱导退保、隐瞒病史等高风险内容直接拦截转人工
- 纯规则实现，**零 LLM 调用**，不增加响应延迟

### 质量保障
- RAGAS 六指标离线评测（检索 4 项 + 生成 2 项），分数回写对应 trace
- Langfuse 线上抽样打分（faithfulness / answer_relevancy）
- 用户对单条回复点赞/点踩，反馈挂到该条消息的 trace
- 183 个单元测试覆盖认证、路由、安全、记忆、评测等核心链路

---

## 系统架构

```
                        ┌─────────────────────────────┐
   React + Ant Design   │  SSE 流式对话 / 会话列表 / 反馈 UI │
                        └──────────────┬──────────────┘
                                       │ HTTP + JWT
                        ┌──────────────▼──────────────┐
                        │        FastAPI (app/)        │
                        │  auth / chat / chat_thread  │
                        │  product / insurance_plan   │
                        └──────────────┬──────────────┘
                                       │
              ┌────────────────────────▼────────────────────────┐
              │           LangGraph 主 Graph（agents/）          │
              │                                                 │
              │   router ──Command──▶ recommendation_plan       │
              │     │                 claim                     │
              │     │                 chitchat                  │
              │     └─────▶ safety ──▶ END / human_handoff      │
              └───────┬───────────────────────┬─────────────────┘
                      │                       │
         ┌────────────▼─────────┐   ┌─────────▼──────────┐
         │  Session Memory      │   │   RAG Pipeline     │
         │  Redis(最近N条+摘要)  │   │  MinerU 解析        │
         │  Store(跨会话画像)    │   │  父子块 + Milvus    │
         └──────────────────────┘   │  BM25+向量+RRF+重排 │
                                    │  Redis L1 缓存      │
                                    └────────────────────┘
                      │
         ┌────────────▼─────────────┐
         │  Langfuse（自托管）       │
         │  trace / 在线评测 / 反馈  │
         └──────────────────────────┘
```

---

## 技术栈

| 层 | 选型 |
|---|---|
| Agent 编排 | LangGraph（StateGraph / Command / interrupt / Store / Checkpointer） |
| 大模型 | DeepSeek（对话）、DashScope `qwen3.7-text-rerank`（精排） |
| RAG | Milvus（稠密 + BM25 混合检索）、RRF 融合、MinerU 文档解析 |
| 存储 | PostgreSQL（业务数据 + Checkpointer）、Redis（缓存 + 短期记忆） |
| 可观测与评测 | Langfuse（trace / score）、RAGAS（六指标离线评测） |
| 后端 | Python 3.14 · FastAPI · SSE · SQLAlchemy · uv |
| 前端 | React · TypeScript · Vite · Ant Design |
| 部署 | Docker Compose（PostgreSQL / Milvus / MinIO / Redis / Langfuse） |

---

## 目录结构

```
monuo-insurance/
├── agent-service/                # 后端
│   ├── app/
│   │   ├── agents/               # LangGraph 编排：路由、三个子 Agent、安全节点、记忆
│   │   ├── rag/                  # 解析、切分、混合检索、重排、检索缓存
│   │   ├── evaluation/           # 在线抽样评测
│   │   ├── modules/              # 业务模块：auth / chat / product / plan ...
│   │   ├── infra/                # 数据库、Checkpointer、Store
│   │   └── core/                 # 配置、日志、可观测性
│   ├── rag_eval/                 # RAGAS 离线评测（题库 / 答题卡 / 判卷 / 分数上传）
│   ├── data/raw/                 # 条款 PDF 与费率表（RAG 原始数据）
│   ├── docs/bad-cases/           # bad-case 复盘文档
│   ├── scripts/                  # 入库脚本
│   └── tests/                    # 183 个单元测试
├── frontend/                     # React 前端
└── deploy/                       # Docker Compose：业务栈 + Langfuse 栈
```

---

## 快速开始

### 1. 准备环境变量

```bash
cd agent-service
cp .env.example .env
# 填入 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / MINERU_TOKEN 等
```

### 2. 启动基础设施

```bash
cd ../deploy
docker compose -f docker-compose.yml up -d      # PostgreSQL + Milvus + Redis
docker compose -f langfuse-compose.yml up -d    # 可选：可观测看板
```

> 镜像默认走轩辕镜像站代理，首次使用需 `docker login docker.xuanyuan.run`；
> 换回官方源时去掉镜像名里的该前缀即可。

### 3. 条款入库

```bash
cd ../agent-service
uv sync
uv run python scripts/ingest_rag.py
```

### 4. 启动后端

```bash
uv run python -m app.main     # http://127.0.0.1:8001
```

### 5. 启动前端

```bash
cd ../frontend
npm install && npm run dev
```

---

## 评测

离线评测复用**生产链路**：逐题真实调用 Agent，捕获工具实际使用的条款片段，再交给 RAGAS 判卷。

```bash
cd agent-service

# 生成答题卡（走真实链路，捕获检索上下文）
uv run python -m rag_eval.run build

# 单题冒烟：确认六个裁判都能出分
uv run python -m rag_eval.run sample --index 0

# 全量六指标判卷，分数回写 Langfuse
uv run python -m rag_eval.run evaluate --metrics full --test-size 1.0 --name full-run
```

| 考谁 | 指标 |
|---|---|
| 检索器 | context_precision / context_recall / context_entity_recall / noise_sensitivity（越低越好） |
| 生成器 | faithfulness / answer_relevancy |

详见 [`agent-service/rag_eval/README.md`](agent-service/rag_eval/README.md)。

### 测试

```bash
cd agent-service
uv run pytest tests/ -q
```

---

## 设计取舍

- **记忆分两层而非一层**：Checkpointer 保留全量原文作为真源，Redis 只做「发给模型的紧凑视图」，摘要失真最多退回窗口截断，不会丢数据。
- **检索缓存只做 L1 精确匹配**：先观察真实命中率再决定是否引入语义缓存，避免为不确定的收益付出复杂度和误命中代价。
- **合规审核不用 LLM**：规则可枚举、可审计、零延迟，也避免用模型判断"自己是否违规"的循环依赖。
- **产品名不让模型猜**：模型只输出产品名，ID 由服务端解析；解析不到就要求澄清，而不是退化成全库检索。

bad-case 复盘见 [`agent-service/docs/bad-cases/`](agent-service/docs/bad-cases/)。

---

## 免责声明

本仓库中的保险条款 PDF 与费率表均来自公开渠道，**仅用于技术学习与研究**，不构成任何保险销售或投保建议。产品名称与条款版权归各自保险公司所有。
