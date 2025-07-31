#!/bin/bash

export PYTHONSTARTUP="spawn"

# 设置模型路径（可以根据需要修改）
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# 设置服务器参数
HOST="0.0.0.0"
PORT="8801"
TRUST_REMOTE_CODE="--trust-remote-code"

# 启动 vLLM 模型服务
echo "启动 vLLM 服务..."
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --host $HOST \
  --port $PORT \
  $TRUST_REMOTE_CODE

