import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TextIteratorStreamer
from threading import Thread

import os 

from .model_zoo import *
from .utils import highlight_text


class Qwen1_5():
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
        self.tokenizer = AutoTokenizer.from_pretrained(modelWeightPath, use_fast=False)
        # 加载模型
        self.model =  AutoModelForCausalLM.from_pretrained(modelWeightPath, torch_dtype=torch.bfloat16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
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

    def chat(self, messages, max_tokens=2048):
        # 调用模型进行对话生成
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=max_tokens)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    def streamChat(self, messages, max_tokens=2048):
        # 流式输出对话
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(self.device)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=max_tokens, streamer=streamer)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        for chunk in response:
            yield chunk
        yield "[DONE]"

    def streamIterChat(self, messages, max_tokens=2048):
        # 流式输出对话 返回队列
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(model_inputs, max_new_tokens=max_tokens, streamer=streamer)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)

        thread.start()
        yield from streamer