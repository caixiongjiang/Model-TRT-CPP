# vllm并发大模型推理
# CUDA 12.1, Pytorch 2.1.2


from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
import uvicorn
import json
import torch

from nets.LLM.model_zoo import *
from nets.LLM.vllm_glm3 import vllmChatGLM3
from nets.LLM.vllm_qwen1_5 import vllmQwen1_5
from utils.utils import return_config
from utils.log import make_log


# 清理GPU内存函数
def torch_gc(device):
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(device):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

def check_content_type(content_type: str = Header(...)):
    if content_type != "application/json":
        raise HTTPException(status_code=415, detail="Unsupported media type, JSON expected")


# 创建FastAPI应用
app = FastAPI()

@app.post("/free_gc")
def free_gc():
    try:
        torch_gc("cuda")
        return {"response": "缓存清理成功"}
    except Exception as e:
        logger.error(f"error: {e}")
        return {"response": "缓存清理失败"}


# LLM chat
@app.post("/LLM/chat")
async def llm_chat(request: Request,
                   app_id: str = Header(None, alias="appId"),
                   request_id: str = Header(None, alias="requestId")):
    try:
        # 检查请求头中的app_id
        if app_id != "LLM_chat":
            raise HTTPException(status_code=400, detail="Invalid appId, {}".format(app_id))
        json_post_raw = await request.json()  # 获取POST请求的JSON数据
        json_post = json.dumps(json_post_raw)  # 将JSON数据转换为字符串
        json_post_list = json.loads(json_post)  # 将字符串转换为Python对象
        logger.info("Request ID: {}, Info message, request info: {}".format(request_id, json_post_list))
        
        # 获取请求中的提示
        prompt = json_post_list.get('prompt')
        if "system_prompt" in json_post_list.keys():
            sys_prompt = json_post_list.get('system_prompt')
        else:
            sys_prompt = ""
        logger.info("Request ID: {}, Info message, system prompt: {}, user prompt: {}".format(request_id, sys_prompt, prompt))

        # 调用模型开始生成
        final_prompt = model.get_prompt(user_message=prompt, system_message=sys_prompt)
        logger.info("Request ID: {}, Info message: 提示词生成成功！".format(request_id))
        if "qwen1.5" in model_params["LLM_MODEL_NAME"]:
            response = model.chat(final_prompt)
        elif "chatglm3" in model_params["LLM_MODEL_NAME"]:
            response = model.chat(final_prompt)

        logger.info("Request ID: {}, Info message: 对话生成成功！".format(request_id))
        logger.info("Request ID: {}, Info message: 大模型回答信息：{}".format(request_id, response))

        answer = {"response": response, "status": 200}

        return answer
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# 主函数入口
if __name__ == '__main__':
    # 加载配置
    log_params, model_params, server_params = return_config()
    # 加载大模型
    if "qwen1.5" in model_params["LLM_MODEL_NAME"]:
        model = vllmQwen1_5(model_params["LLM_MODEL_NAME"])
        # 先推理一遍
        model.chat(messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "你好"}
            ])
    elif "chatglm3" in model_params["LLM_MODEL_NAME"]:
        model = vllmChatGLM3(model_params["LLM_MODEL_NAME"]) 
        # 先推理一遍
        model.chat(messages="""<|system|>You are a helpful assistant.<|user|>你好<|assistant|>""")
    else:
        raise ValueError("Model name should be in {}".format(llm_weight_zoo.keys())) 

    # 开启日志实例
    logger = make_log(log_params)
    logger.info("Info message: server is loading...")
    uvicorn.run(app, host=server_params["IP"], port=server_params["PORT"], workers=1)  # 在指定端口和主机上启动应用