# vLLM压测脚本
import requests
import json
import time
import threading

sys_prompt = """
    Answer the following questions as best as you can. You have access to the following information:
    [
        {enterprise_info}
    ]
    if related information is not available, please respond with "我们目前暂时没有记录该信息". 
    """

prompt = """{question}是什么？"""


def post_stream(model_name, sys_prompt, prompt, url, timeout=10):
    
    stream_body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "stream": True
    }

    response = requests.post(url, json=stream_body, stream=True, timeout=timeout)
    return response

# 函数来模拟用户发送请求
def send_request(user_id, model_name, sys_prompt, prompt, url, timeout=10):

    stream_body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "stream": True
    }

    start_time = time.time()
    response = requests.post(url, json=stream_body, stream=True, timeout=timeout)
    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            data = json.loads(line_str.split(": ", 1)[1])
            if data["choices"][0]["finish_reason"] == "stop":
               break
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"User {user_id}: Status Code - {response.status_code}, Response Time - {elapsed_time:.2f} seconds")



def analy_openai_stream(response):
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


if __name__ == "__main__":
    enterprise_info = {
        "公司信息": "亚信科技控股公司",
        "注册时间": "2014年9月",
        "注册资金": "5000万",
        "上次拜访时间": "2015年10月"
    }

    question = "上次拜访时间"

    sys_prompt = sys_prompt.format(enterprise_info=enterprise_info)
    prompt = prompt.format(question=question)
    timeout = 30
    model_name = "qwen1.5-4b-chat"
    url = "http://localhost:10005/v1/chat/completions"

    # 测试回答效果
    try: 
        response = post_stream(model_name=model_name, sys_prompt=sys_prompt, prompt=prompt, url=url, timeout=timeout)
        analy_openai_stream(response)
    except requests.exceptions.Timeout:
        print("大模型请求超时")


    # 压测
    CONCURRENT_USERS = 10 # 用户数
    TOTAL_REQUESTS = 100 # 总请求数

    threads = []
    for i in range(CONCURRENT_USERS):
        for j in range(TOTAL_REQUESTS // CONCURRENT_USERS):
            thread = threading.Thread(target=send_request, args=(i * TOTAL_REQUESTS // CONCURRENT_USERS + j, 
                                                                 model_name, 
                                                                 sys_prompt, 
                                                                 prompt, 
                                                                 url, 
                                                                 timeout))
            threads.append(thread)

    start_time = time.time()
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # 计算总共花费的时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"用户数：{CONCURRENT_USERS}， 总并发数：{TOTAL_REQUESTS}")
    print(f"Total Time: {total_time:.2f} seconds")
