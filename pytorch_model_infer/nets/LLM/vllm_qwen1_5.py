import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from threading import Thread

import os 

from .model_zoo import *
from .utils import highlight_text

# TODO：统一为openai接口的返回形式

class vllmQwen1_5():
    def __init__(self, modelName="qwen1.5-4b-chat") -> None:
        # 检查是否支持模型
        assert modelName in llm_weight_zoo, "modelName should be in {}".format(llm_weight_zoo.keys())
        modelWeightPath = llm_weight_zoo[modelName]
        # 检查本地是否下载模型
        assert os.path.exists(modelWeightPath), "Model weight file: {}".format(modelWeightPath) + \
                                                " does not exist.\n Please download {} ".format(modelName) + \
                                                "model in" + \
                                                highlight_text("'{}'".format(llm_model_url_zoo[modelName]))
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(modelWeightPath)
        # 加载模型 Tesla T4需要使用float16 tokenizer=None会默认使用配套的标记器
        self.model = LLM(model=modelWeightPath, tokenizer=None, 
                         dtype="auto", trust_remote_code=True,
                         max_model_len=2048)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = modelName

    def get_prompt(self, user_message, system_message):
        if system_message == "":
            system_message = "You are a helpful assistant."

        assert user_message != None, "我是{}大模型,你必须输入提问问题".format(self.model_name)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        return messages

    def chat(self, messages, max_tokens=2048, temperature=0.9, top_p=0.8):
        # 调用模型进行对话生成
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        output = self.model.generate([input_ids], sampling_params)[0]
        response = output.outputs[0].text

        return response


