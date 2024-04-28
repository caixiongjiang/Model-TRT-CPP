import requests
import json

headers = {
    "Content-Type": "application/json"
}

def post(url, json, headers):
    response = requests.post(url, json=json, headers=headers)
    return response

def streamPost(url, json, headers):
    response = requests.post(url, json=json, headers=headers, stream=True)
    return response

if __name__ == "__main__":
    ip = "localhost"
    port = 10005
    

    sys_prompt = """
    你现在是一个移动产品推荐专家，我将告诉你产品名称、产品优势信息，你需要理解以上信息，你需要理解以上信息，以第一人称的方式生成一段供营销人员使用的产品推荐话术。生成的内容需要严格按照我指定的规则。
    生成规则：
    1、生成的格式需要完全按照下方样例的格式。
    2、生成的推荐话术里，开头统一称呼老板您好，推荐话术长度严格限定在200字以内。
    3、生成的推荐话术严格限制只能使用我提供给你的信息，严禁生成额外信息。
    4、生成的推荐话术不能只是使用我提供给你的信息进行拼接，需要使用我提供的信息进行精简、润色生成最终的推荐话术。
    生成样例：
    1、老板您好，记得您提过对灵活高效的工作设备有兴趣。我推荐“云电脑2.0”，它覆盖所有主流平台，无论Windows、Mac还是手机，都能无缝切换。就像把办公室随身携带，而且外设兼容性好，4K画质体验超棒。安装应用简单，400多个生态应用任您选，还有实时安全防护。开机即用，省钱又省心，运维管理也集中，安全性高，成本只有传统电脑的1/3。如果想深入了解，我们可以安排在您空的时间进行详细讲解，怎么样？
    2、老板您好，根据咱们之前的交流，您对高效安全的视频会议系统有需求。我推荐“云视讯”。它全场景覆盖，无论会议室、办公室还是远程，都能轻松应对。全球接入，海外同事也不愁。高清音视频，定制化终端，满足各类需求。而且，它通过了高级安全认证，确保会议隐私。更重要的是，它能承载大量并发会议，上千人同时参会也不在话下。如果感兴趣，我们可以安排在您空的时间，由专家为您详细解读并定制专属方案，怎么样？
    """
    
    prompt = """
    产品名称：【商务专线】。
    产品优势信息：【"1.支持静态IP的大宽带网络接入
可根据企业用户对静态IP的需求，提供更加流畅的上网体验。
2.提供一站式产品方案
以商务快线网络接入为基础、企业网关硬件为载体，解决公共WiFi、语音通话、公播音乐、电视娱乐、屏幕定制、信息安全防护、视频监控、企业办公、云应用等信息化需求，面向不同的细分场景提供一站式通信服务解决方案"】。请为该产品生成产品推荐话术，字数限制在200字以内。
    """

    body = {
        "model": "qwen1.5-4b-chat",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    stream_body = {
        "model": "qwen1.5-4b-chat",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "stream": True
    }

    url = f"http://{ip}:{port}/v1/chat/completions"
    
    # 发送 POST 请求
    response = post(url, json=body, headers=headers)

    # 解析响应
    if response.status_code == 200:
        print("Return:", response.text)
    else:
        print("Error:", response.text)

    # 发送stream Post 请求
    response = streamPost(url, json=stream_body, headers=headers)
    full_text = ""
    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            print(line_str)
            data = json.loads(line_str.split(": ", 1)[1])
            if data["choices"][0]["finish_reason"] != "stop":
                if data["choices"][0]["delta"].get("content") is not None:
                    full_text += data["choices"][0]["delta"]["content"]
                else:
                    continue
            else:
                break
    print("大模型完整回答：\n", full_text)