"""RAG 评测 CLI。

用法（在项目根目录执行）：
  python -m rag_eval.run build [--limit N] [--rebuild]    # 生成答题卡
  python -m rag_eval.run evaluate [--test-size 0.3] [--metrics fast|full] [--name xxx]
  python -m rag_eval.run sample [--index 0]               # 单题跑全 6 指标（冒烟）

前置条件：
  - Milvus / PostgreSQL 可访问（答题卡走真实 Agent）
  - Ollama 已启动并拉取 qwen3-embedding:0.6b（判卷向量）
  - .env 中配置 DEEPSEEK_API_KEY
"""

import argparse
import asyncio
import selectors
import sys

import rag_eval.compat  # noqa: F401  必须先于 ragas 加载

# 注意：dataset / evaluate 采用函数内惰性导入。
# 因为 dataset 会导入生产 Agent（进而初始化 Milvus 连接），
# 放在模块顶层会导致没启动 Milvus 时连 `--help` 都跑不了。

# Windows 控制台默认 GBK，输出中文/符号会崩，统一改 UTF-8
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_async(coro):
    """Windows 下 psycopg 异步连接池不支持 ProactorEventLoop，统一用 Selector"""
    loop_factory = (
        (lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()))
        if sys.platform == "win32"
        else None
    )
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        return runner.run(coro)


def cmd_build(args) -> None:
    from rag_eval.dataset import build_dataset

    run_async(build_dataset(limit=args.limit, rebuild=args.rebuild))


def cmd_evaluate(args) -> None:
    from rag_eval.dataset import build_dataset, load_cached_dataset
    from rag_eval.evaluate import build_judges, run_experiment

    dataset = load_cached_dataset()
    if dataset is None:
        print("没有缓存答题卡，先现场答题生成...")
        dataset = run_async(build_dataset(limit=args.limit))

    build_judges()
    run_async(
        run_experiment(
            dataset,
            test_size=args.test_size,
            metrics=args.metrics,
            name=args.name,
        )
    )


def cmd_sample(args) -> None:
    from rag_eval.dataset import load_cached_dataset
    from rag_eval.evaluate import build_judges, score_single

    dataset = load_cached_dataset()
    if dataset is None:
        raise SystemExit("没有缓存答题卡，请先运行: python -m rag_eval.run build")

    build_judges()
    run_async(score_single(dataset, index=args.index))


def main() -> None:
    parser = argparse.ArgumentParser(description="保险条款 RAG 评测（RAGAS）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="生成答题卡（调 Agent 答题）")
    p_build.add_argument("--limit", type=int, default=None, help="只答前 N 题")
    p_build.add_argument("--rebuild", action="store_true", help="丢弃缓存强制重答")
    p_build.set_defaults(func=cmd_build)

    p_eval = sub.add_parser("evaluate", help="批量判卷出成绩单")
    p_eval.add_argument("--limit", type=int, default=None, help="（无缓存时）只答前 N 题")
    p_eval.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="判卷抽样比例（默认 0.3）",
    )
    p_eval.add_argument(
        "--metrics",
        choices=["fast", "full"],
        default="fast",
        help="fast=2核心指标；full=全6指标（更慢）",
    )
    p_eval.add_argument("--name", default="baseline", help="实验名（成绩单目录名）")
    p_eval.set_defaults(func=cmd_evaluate)

    p_sample = sub.add_parser("sample", help="单题跑全 6 指标（批量前冒烟）")
    p_sample.add_argument("--index", type=int, default=0, help="答题卡行号")
    p_sample.set_defaults(func=cmd_sample)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
