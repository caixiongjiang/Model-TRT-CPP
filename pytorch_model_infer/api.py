from fastapi import FastAPI, Body, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import torch
import io
import base64


from nets.classification.model_infer import classifyModel
from nets.detection.yolo_infer import YOLO_Inference
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
async def detection_predict(input_data: ImageInput = Body(...),
                            app_id: str = Header(None, alias="appId"),
                            request_id: str = Header(None, alias="requestId")):
    try:
        # 检查请求头中的app_id
        if app_id != "detection_infer":
            raise HTTPException(status_code=400, detail="Invalid appId, {}".format(app_id))
        logger.info("Request ID: {}, Info message, request info: {}".format(request_id, input_data))
        # 解析得到文件地址
        file_path = input_data.file_path
        logger.info("Request ID: {}, Info message, file_path: {}".format(request_id, file_path))
        # 图片前处理
        image_tensor = detection_model.preprocess(imagePath=file_path, letter_box=True)
        logger.info("Request ID: {}, Info message: The pre-processing of the picture is completed.".format(request_id))
        # 图片推理
        prediction = detection_model.inference(image_tensor)
        logger.info("Request ID: {}, Info message: Image inference was completed.".format(request_id))
        # 后处理
        image, labels, scores, boxes = detection_model.postprocess(prediction, conf=0.5, iou=0.7, save=False)
        logger.info("Request ID: {}, Info message: Image postprocess was completed.".format(request_id))
        # 图像转化为base64
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()
        img_base64 = base64.b64encode(img_byte_array).decode('utf-8')
        logger.info("Request ID: {}, Info message: Convert the image to base64.".format(request_id))
        # 整合检测框信息
        predict_results = detection_model.gen_predict_results(labels, scores, boxes)
        logger.info("Request ID: {}, Info message: Complete the integration of detection box information.".format(request_id)) 

        returnInfo = { "results": predict_results["results"],
                       "output": img_base64
                    }   

        return JSONResponse(content=returnInfo, status_code=200)
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
    # 目前暂时使用命令行来控制使用的gpu id（单卡）
    log_params, model_params, server_params = return_config()
    # 模型初始化
    classification_model = classifyModel(modelName=model_params["CLASSIFY_MODEL_NAME"])
    detection_model = YOLO_Inference(modelName=model_params["DETECT_MODEL_NAME"])
    # segmentation_model = None
    # 先使用模型推理一遍
    classification_model.inference(imageTensor=torch.randn(1, 3, 224, 224).cuda())
    detection_model.inference(image_tensor=torch.randn(1, 3, 640, 640).cuda())

    logger = make_log(log_params)
    logger.info("Info message: server is loading...")
    uvicorn.run(app, host=server_params["IP"], port=server_params["PORT"])

