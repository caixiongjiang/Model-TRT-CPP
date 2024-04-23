import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

import os 

from .model_zoo import *
from .utils import * 


class glm3():
    def __init__(self, modelName="chatglm3-6b", vllm=False) -> None:
        # 检查是否支持模型
        assert modelName in llm_weight_zoo, "modelName should be in {}".format(llm_weight_zoo.keys())
        modelWeightPath = llm_weight_zoo[modelName]
        # 检查本地是否下载模型
        assert os.path.exists(modelWeightPath), "Model weight file: {}".format(modelWeightPath) + \
                                                " does not exist.\n Please download {}".format(modelName) + \
                                                "model in" + \
                                                highlight_text("'{}'".format(llm_model_url_zoo[modelName]))
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(modelWeightPath, trust_remote_code=True)
        # 加载模型
        self.model =  AutoModelForCausalLM.from_pretrained(modelWeightPath, trust_remote_code=True).half()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_name = modelName

    def get_prompt(self, user_message, system_message):
        if system_message == "":
            system_message = "You are an artificial intelligence assistant."
        messages = """
        <|system|>
        {}
        <|user|>
        {}
        <|assistant|>
        """.format(system_message, user_message)

        return messages
        

    def chat(self, messages, max_length=None, top_p=None, temperature=None, history=[]):
        # 调用模型进行对话生成
        response, history = self.model.chat(
            self.tokenizer,
            messages,
            history=history,
            max_length=max_length if max_length else 2048,  # 如果未提供最大长度，默认使用2048
            top_p=top_p if top_p else 0.7,  # 如果未提供top_p参数，默认使用0.7
            temperature=temperature if temperature else 0.95  # 如果未提供温度参数，默认使用0.95
        )

        return response, history
    
    def streamChat(self, messages, max_length=None, top_p=None, temperature=None, history=None):
        
        response, history = self.model.stream_chat(self.tokenizer, 
                                                   messages, 
                                                   history=history,
                                                   max_length=max_length if max_length else 2048,  # 如果未提供最大长度，默认使用2048
                                                   top_p=top_p if top_p else 0.7,  # 如果未提供top_p参数，默认使用0.7
                                                   temperature=temperature if temperature else 0.95)  # 如果未提供温度参数，默认使用0.95

        return response


    def streamIterChat(self, messages, max_length=None, top_p=None, temperature=None, history=None):
        size = 0
        response = ""
        for response, history in self.model.stream_chat(self.tokenizer, 
                                                        messages, 
                                                        history=history,
                                                        max_length=max_length if max_length else 2048,
                                                        top_p=top_p if top_p else 0.7,
                                                        temperature=temperature if temperature else 0.95):
            chunk = response[size:]
            history = [list(h) for h in history]
            size = len(response)
            yield chunk
