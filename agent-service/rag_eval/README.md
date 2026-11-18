# rag_eval：保险条款 RAG 评测

复用生产链路的 Agent + 检索器，输出 RAGAS 六指标成绩单，用来量化评估
「混合检索 + RRF + 重排」和查询优化改动的实际效果。

## 评测的是哪条链路

```
题库问题
   ↓
生产 Agent（意图识别 → 保险顾问子Agent）
   ↓
retrieve_insurance_clause 工具
   ↓  稠密 + BM25 → RRF融合 → 云端重排 → 父块回表
条款上下文 ──捕获──┐
   ↓              │
模型生成回答       │
   ↓              │
答题卡：user_input / response / retrieved_contexts / reference / trace_id
   ↓
RAGAS 判卷（DeepSeek 判文本 + 本地 Ollama 算向量）
   ↓
六指标成绩单 → rag_eval/results/experiments/<实验名>.csv
   ↓
分数回写 Langfuse → 每道题的 trace 详情页 + Scores 看板
```

上下文捕获通过 `app/rag/capture.py` 的 contextvar 实现：**生产环境未开启时是空操作**，
只有评测脚本在调用 Agent 前用 `capture_contexts()` 开启才会记录。

## 前置条件

| 依赖 | 用途 |
|------|------|
| Milvus + PostgreSQL 可访问 | 答题卡会真实调用 Agent 和检索 |
| `DEEPSEEK_API_KEY` | 答题 + 判卷 LLM |
| Ollama + `ollama pull qwen3-embedding:0.6b` | 判卷向量（answer_relevancy；fast 档不需要） |
| Langfuse（可选） | 回写评测分数；不配置时判卷照常跑，只跳过上传 |

`ragas` 已在主依赖中（`uv sync` 即可）。

## 使用步骤

### 1. 准备题库

编辑 `resources/insurance_eval_dataset.json`，逐题填写 `ground_truth`：

```json
{
  "question": "达尔文12号重大疾病保险的等待期是多久？",
  "product_name": "达尔文12号重大疾病保险",
  "ground_truth": "等待期为180天，自合同生效之日起计算。等待期内因非意外原因确诊重疾，退还已交保费，合同终止。"
}
```

> `ground_truth` 必须从条款原文提炼——它是 Context Recall 的评分尺子，
> 写错会冤枉检索器。缺任何一题的 `ground_truth`，脚本会直接报错并提示行号。

### 2. 生成答题卡

```bash
uv run python -m rag_eval.run build              # 全部题目
uv run python -m rag_eval.run build --limit 2    # 先冒烟
uv run python -m rag_eval.run build --rebuild    # 强制重答
```

答题卡缓存在 `results/datasets/insurance_rag_eval.csv`，改题库后需 `--rebuild`。

启用 Langfuse 时（`.env` 中 `LANGFUSE_ENABLED=true`），每道题都会生成独立 trace，
并把 `trace_id` 写入答题卡新增的 `trace_id` 列；判卷阶段据此回写分数。
输出里每题会标注 `trace: 有/无`。

### 3. 先单题冒烟

```bash
uv run python -m rag_eval.run sample --index 0
```

逐项打印 6 个指标，确认裁判都能正常出分（避免批量跑一半发现 NaN）。
注意：`sample` 只做调试，**不上传分数**。

### 4. 批量判卷

```bash
# 全 6 指标，判全部题目（完整评测用这条）
uv run python -m rag_eval.run evaluate --metrics full --test-size 1.0 --name full-run

# 2 个核心指标，快
uv run python -m rag_eval.run evaluate --metrics fast

# 抽样判卷（8 题题库判 4 题）+ 自定义实验名
uv run python -m rag_eval.run evaluate --metrics full --test-size 0.5 --name half-run
```

成绩单落在 `results/experiments/<实验名>.csv`（注意是 CSV 文件，不是目录）。
判卷过程中每个指标算完即回写 Langfuse（上传失败只告警，不影响判卷）。

### 5. 在 Langfuse 查看评测分数

打开 Langfuse 看板（即 `.env` 中 `LANGFUSE_HOST` 配置的地址，如 `http://localhost:3100`）：

- **Scores** 页可按指标名筛选，comment 为 `offline_eval` 的即离线评测分数
- 点任意分数可跳到对应题目的 trace，直接看到问题、检索到的条款、AI 回答
- 离线分数（`offline_eval`）与线上采样评测（`online_eval`）共用同名指标，可统一看趋势

前提是答题卡带 `trace_id`；旧缓存答题卡没有该列，需 `build --rebuild` 重答一次。

## 六指标含义

| 考谁 | 指标 | 方向 |
|------|------|------|
| 检索器 | context_precision | 越高越好 |
| 检索器 | context_recall | 越高越好 |
| 检索器 | context_entity_recall | 越高越好 |
| 检索器 | noise_sensitivity | **越低越好** |
| 生成器 | faithfulness | 越高越好 |
| 生成器 | answer_relevancy | 越高越好 |

## 做 A/B 对比的建议

每次改动（RRF、重排模型、查询改写、术语扩展）都先 `--rebuild` 生成新答题卡，
再用不同的 `--name` 跑判卷，对比两次成绩单：

```bash
uv run python -m rag_eval.run build --rebuild
uv run python -m rag_eval.run evaluate --name rrf-plus-rerank
```

同一套题库、同一批裁判，指标才可比。

## 已知坑

- **判卷 NaN**：DeepSeek 是混合思考模型，判卷已显式关闭思考并把 `max_tokens` 提到 8192
- **限流**：答题间隔见 `config.REQUEST_DELAY`，报限流就调大
- **检索为空**：若某题 `retrieved_contexts` 为空（Agent 没调用检索工具），
  Context Recall 会低——这本身就是有价值的信号，先看题目是否让模型误判了意图
- **旧答题卡没有 trace_id**：分数上传会被跳过并告警，用 `build --rebuild` 重答即可
- **ragas 遥测**：默认会向 `t.explodinggradients.com` 上报（每次超时约 1 秒），
  脚本已设置 `RAGAS_DO_NOT_TRACK=true` 关闭
- **Windows 事件循环**：psycopg 异步连接不支持 ProactorEventLoop，`run.py` 已统一切到
  SelectorEventLoop，无需手工处理
