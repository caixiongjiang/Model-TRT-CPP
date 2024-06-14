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
    你好
    """
    
    prompt = """
    请给我讲一个故事
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