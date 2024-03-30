# TODO: 完善日志模块
from fastapi import FastAPI, Body, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import torch


from nets.classification.model_infer import classifyModel
from utils.utils import *
from utils.log import make_log

app = FastAPI()


class ImageInput(BaseModel):
    file_path: str


def check_content_type(content_type: str = Header(...)):
    if content_type != "application/json":
        raise HTTPException(status_code=415, detail="Unsupported media type, JSON expected")


# 分类任务图片推理接口
@app.post("/classification/predict")
async def classification_predict(input_data: ImageInput = Body(...),
                                 app_id: str = Header(None, alias="appId"),
                                 request_id: str = Header(None, alias="requestId")):

    try:
        # 检查请求头中的app_id
        if app_id != "classification_infer":
            raise HTTPException(status_code=400, detail="Invalid appId, {}".format(app_id))
        logger.info("Request ID: {}, Info message, request info: {}".format(request_id, input_data))
        # 解析得到文件地址
        file_path = input_data.file_path
        logger.info("Request ID: {}, Info message, file_path: {}".format(request_id, file_path))    
        # 图片前处理
        image_tensor = classification_model.preprocess_image(file_path)
        logger.info("Request ID: {}, Info message: The pre-processing of the picture is completed.".format(request_id))   
        # 图片推理
        prediction = classification_model.inference(image_tensor)
        logger.info("Request ID: {}, Info message: Image inference was completed.".format(request_id))
        # 后处理
        prediction = classification_model.postprocess(prediction, "./data/classname/imagenet1k.names")
        logger.info("Request ID: {}, Info message: Image postprocess was completed.".format(request_id))

        returnInfo = {"prediction": prediction}

        logger.info("Request ID: {}, Info message, return info: {}".format(request_id, returnInfo))

        return JSONResponse(content=returnInfo, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

# 检测任务图片推理接口
@app.post("/detection/predict")
async def detection_predict(input_data: ImageInput = Body(...)):
    try:
        return JSONResponse(content={"prediction": "detection"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# 分割任务图片推理接口
@app.post("/segmentation/predict")
async def segmentation_predict(input_data: ImageInput = Body(...)):
    try:
        return JSONResponse(content={"prediction": "segmentation"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

if __name__ == "__main__":
    # 目前暂时使用命令行来控制使用的gpu id
    # 同时开启分类，分割，检测三个模型的推理服务
    log_params, model_params, server_params = return_config()
    # 模型初始化
    classification_model = classifyModel(modelName=model_params["CLASSIFY_MODEL_NAME"])
    # detection_model = None
    # segmentation_model = None
    # 先使用模型推理一遍
    classification_model.inference(imageTensor=torch.randn(1, 3, 224, 224).cuda())

    logger = make_log(log_params)
    logger.info("Info message: server is loading...")
    uvicorn.run(app, host=server_params["IP"], port=server_params["PORT"])

