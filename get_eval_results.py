import os
import argparse
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description="Run model evaluation with specified model, datasets, and max_tokens.")
parser.add_argument(
    '--model',
    type=str,
    required=True,
    help='The model name to be used for the task, e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B".'
)
parser.add_argument(
    '--dataset',
    type=str,
    required=True,
    help='The maximum number of tokens to generate, e.g., 2048.'
)

# 解析命令行参数
args = parser.parse_args()

# 使用命令行传入的模型名称、数据集和max_tokens
task_config = TaskConfig(
    model=args.model,  # 使用命令行传入的模型称
    datasets=[args.dataset],  # 使用命令行传入的数据集列表
    eval_batch_size=32,  # 发送请求的并发数
    work_dir=args.model,
    judge_strategy='llm',
    judge_model_args={
        'model_id': 'Qwen/Qwen2.5-7B-Instruct',
        'api_url': 'https://api-inference.modelscope.cn/v1',
        'api_key': 'ms-f5432315-aa7f-418c-a40c-ec50f78b7b72',
    },
    generation_config={
        'temperature': 0.6,  # 采样温度 (deepseek 报告推荐值)
        'top_p': 0.95,  # top-p采样 (deepseek 报告推荐值)
        'n': 1,  # 每个请求产生的回复数量
    },
)

# 运行任务
run_task(task_config)
