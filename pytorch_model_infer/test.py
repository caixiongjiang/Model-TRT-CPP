import timm 
import requests
import uuid 


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

    # 上传的图片文件路径
    body = {
        "prompt": "你好,请给我讲一个故事，随便编一个",
        "system_prompt": ""
    }

    stream_url = "http://localhost:10005/LLM/streamChat"
    # 发送 POST 请求
    stream_response = streamPost(stream_url, json=body, headers=headers)

    for line in stream_response.iter_lines():
        if line:
            print(line.decode("utf-8"))

    url = "http://localhost:10005/LLM/chat"
    # 发送 POST 请求
    response = post(url, json=body, headers=headers)

    # 解析响应
    if response.status_code == 200:
        print("Return:", response.text)
    else:
        print("Error:", response.text)
