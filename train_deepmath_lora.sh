#!/bin/bash
set -x  # 打印每条执行命令

# ============================
# 模型路径
# ============================
MODEL_PATH="/data1/hf-models/Qwen3/Qwen3-0.6B"

# ============================
# 数据路径
# ============================
TRAIN_FILE="/data1/datasets/deepmath-sampled/train.parquet"
TEST_FILE="/data1/datasets/deepmath-sampled/test.parquet"

LOG_DIR="./logs"
mkdir -p $LOG_DIR

# ============================
# 训练配置
# ============================
TRAIN_BATCH_SIZE=1250
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=20480
OVERLONG_BUFFER_LENGTH=4096
OVERLONG_PENALTY_FACTOR=1.0 
MAX_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

SAVE_FREQ=4
TEST_FREQ=-1
EPOCHS=10
GPUS=$(nvidia-smi -L | wc -l)
GPUS=2
PROJECT="entropy-token-less"
EXP="deepmath"

# ============================
# 启动训练
# ============================
python main_edlao.py \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.overlong_buffer_length=$OVERLONG_BUFFER_LENGTH \
    data.overlong_penalty_factor=$OVERLONG_PENALTY_FACTOR \
    data.overlong_enable=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.entropy_coeff_annealing=cosine \
    actor_rollout_ref.actor.use_entropy_advantage=True \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_advantage_alpha=0.4 \
    actor_rollout_ref.actor.entropy_advantage_kappa=2.0 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_LEN \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra'] \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_LEN \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_LEN \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_LEN \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    length_rewards.use_length_reward=True \
    length_rewards.reward_scale=0.5 \
    custom_reward_function.name=deepmath_reward_fn \
    custom_reward_function.path=utils/deepmath_reward.py \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT \
    trainer.experiment_name=$EXP \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$EPOCHS \
    2>&1 | tee $LOG_DIR/$EXP.log