import timm 
import requests
import uuid 


headers = {
    "appId": "classification_infer",
    "requestId": str(uuid.uuid4())
}


if __name__ == "__main__":
    # model = timm.create_model('resnet18.a3_in1k')
    # print(model.default_cfg)
    # model = timm.create_model('resnet34.a3_in1k')
    # print(model.default_cfg)
    # model = timm.create_model('resnet50.a3_in1k')
    # print(model.default_cfg)
    # model = timm.create_model('resnet101.a3_in1k')
    # print(model.default_cfg)
    # model = timm.create_model('resnet152.a3_in1k')
    # print(model.default_cfg)

    # 上传的图片文件路径
    body = {
        "file_path": "./data/images/cat.jpg"
    }
    # 发送 POST 请求
    response = requests.post("http://localhost:10005/classification/predict", json=body, headers=headers)
    # 解析响应
    if response.status_code == 200:
        print("Return:", response.text)
    else:
        print("Error:", response.text)
