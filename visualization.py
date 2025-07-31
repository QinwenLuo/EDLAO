import os
import argparse
from evalscope.third_party.thinkbench import run_task

# 使用 argparse 解析命令行参数
parser = argparse.ArgumentParser(description="Run evaluation task")
parser.add_argument('--report_path', type=str, required=True, help="Path to store the model inference results")
parser.add_argument('--api_url', type=str, default='http://0.0.0.0:8801/v1', help="Inference service URL")
parser.add_argument('--model_name', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', help="Model name")
parser.add_argument('--max_tokens', type=int, default=20000, help="Maximum tokens for filtering")
parser.add_argument('--count', type=int, default=200, help="Number of outputs to select per subset")

# 解析命令行参数
args = parser.parse_args()

# 配置评测服务
judge_config = dict(  # 评测服务配置
    api_key='EMPTY',
    base_url=args.api_url,  # 使用从命令行传入的推理服务地址
    model_name='DeepSeek-R1-Distill-Qwen-7B',  # 使用从命令行传入的模型名称
)

# 配置模型信息
model_config = dict(
    report_path=args.report_path,  # 使用从命令行传入的报告路径
    model_name=args.model_name,  # 模型名称
    tokenizer_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',  # 模型tokenizer路径，用于计算token数量
    dataset_name='math_500',  # 数据集名称
    subsets=['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],  # 数据集子集
    split_strategies='separator',  # 推理步骤分割策略
    judge_config=judge_config  # 将judge_config传入
)

# 配置筛选参数
max_tokens = args.max_tokens  # 从命令行传入的max_tokens
count = args.count  # 从命令行传入的count

# 评测模型思考效率
run_task(model_config, output_dir='outputs', max_tokens=max_tokens, count=count)

