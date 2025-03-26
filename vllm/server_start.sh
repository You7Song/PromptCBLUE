export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python vllm/api_server.py \
--trust_remote_code True \
--model ./models/llama_lora \
--tensor-parallel-size 1 \
--served-model-name Llama3_8B \
# --gpu-memory-utilization 0.4 \
# --dtype bfloat16 \