### OpenAI接口

* vllm实现openai格式的接口（流式&&非流式）:
```bash
# --type 数据类型，一般来说直接不指定，默认使用auto，但是部分显卡不支持一些数据类型的计算
# --max-num-seqs 每次迭代的最大序列数，默认使用256
# --gpu-memory-utilization gpu内存利用率，默认使用0.9
# --enforce-eager 始终使用渴望模式PyTorch。如果False，将在混合中使用渴望模式和CUDA图，以获得最大的性能和灵活性。(gpu显存不够用的时候强制执行，但是速率会变慢)
# --max-model-len 模型上下文的最大长度
# --enable-lora 是否允许lora微调权重
# --lora-dtype LoRA权重的数据类型
python3 -m vllm.entrypoints.openai.api_server \
--model ./data/model_zoo/qwen1.5-4b-chat \
--served-model-name qwen1.5-4b-chat \
--host 0.0.0.0 \
--port 10005 \
--max-model-len 2048 \
--dtype half \
--max-num-seqs 256 \
--gpu-memory-utilization 0.9 \
--enforce-eager True \
--enable-lora True \
--lora-dtype "auto" 
```

* Transformers原生接口（流式&&非流式）：

```bash
python3 chat_api.py
```