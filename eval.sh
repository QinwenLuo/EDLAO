#!/bin/bash

# exec &> /dev/null

GPU_LIST=(0 1 2 3)


dataset_list=(
	'math_500'
	'aime24'
	'aime25'
 	'gpqa'
)


max_tokens_list=(
	20000
)


model_list=(
	deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
)


task=0

check_gpu_availability() {
    local gpu_device=$1
    local memory_threshold_percentage=25  # Set your desired memory threshold percentage

    local gpu_info_total=$(nvidia-smi --id=$gpu_device --query-gpu=memory.total --format=csv,noheader,nounits)
    local gpu_info_used=$(nvidia-smi --id=$gpu_device --query-gpu=memory.used --format=csv,noheader,nounits)
    local total_memory=$(echo $gpu_info_total | awk -F ',' '{print $1}')
    local used_memory=$(echo $gpu_info_used | awk -F ',' '{print $1}')
    local memory_threshold=$((total_memory * memory_threshold_percentage / 100))

    if [[ $((total_memory - used_memory)) -ge $memory_threshold ]]; then
        return 0  # GPU has sufficient memory available
    else
        return 1  # GPU does not have sufficient memory available
    fi
}



for dataset in "${dataset_list[@]}"; do
  for max_tokens in "${max_tokens_list[@]}"; do
      GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}

      while ! check_gpu_availability $GPU_DEVICE; do
        sleep 60
        let "task=task+1"
        GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
      done

      CUDA_VISIBLE_DEVICES=$GPU_DEVICE python get_eval_results.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/ --max_tokens $max_tokens --dataset $dataset &

      sleep 1
      let "task=task+1"
  done
done


