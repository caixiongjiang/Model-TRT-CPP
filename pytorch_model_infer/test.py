import timm 
import requests
import uuid 
import json

from utils.utils import return_config

headers = {
    "appId": "LLM_chat",
    "requestId": str(uuid.uuid4())
}

def post(url, json, headers):
    response = requests.post(url, json=json, headers=headers)
    return response

def streamPost(url, json, headers):
    response = requests.post(url, json=json, headers=headers, stream=True)
    return response

if __name__ == "__main__":
    # model = timm.create_model('vit_tiny_r_s16_p8_224.augreg_in21k_ft_in1k')
    # print(model.default_cfg)
    # model = timm.create_model('vit_small_r26_s32_224.augreg_in21k_ft_in1k')
    # print(model.default_cfg)
    # model = timm.create_model('vit_large_r50_s32_224.augreg_in21k_ft_in1k')
    # print(model.default_cfg)

    log_params, model_params, server_params = return_config()
    ip = "localhost"
    port = server_params["PORT"]
    llm_model_name = server_params["LLM_MODEL_NAME"]

    sys_prompt = """"""
    
    prompt = """你好，请给我讲一个故事，随便编一个"""

    # 请求体
    llm_body = {
        "prompt": prompt,
        "system_prompt": sys_prompt
    }

    stream_url = f"http://{ip}:{port}/LLM/streamChat"
    # 发送 POST 请求
    stream_response = streamPost(stream_url, json=llm_body, headers=headers)
    
    full_text = ""
    for line in stream_response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            print(line_str)
            data = json.loads(line_str.split(": ", 1)[1])
            if data["data"] != "[DONE]":
                full_text += data["data"]
    print(f"{llm_model_name} 大模型完整回答：\n", full_text)

    url = f"http://{ip}:{port}/LLM/chat"
    # 发送 POST 请求
    response = post(url, json=llm_body, headers=headers)

    # 解析响应
    if response.status_code == 200:
        print(f"{llm_model_name} 大模型回答：\n", response.text)
    else:
        print("Error:", response.text)
