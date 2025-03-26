CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --model ./models/llama_lora \
    --served-model-name Llama3_8B \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 900 \
    --dtype=half \
    # --gpu-memory-utilization 0.8 \
    # --enable-lora