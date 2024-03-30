# TDDO: 完善api请求测试脚本
import timm 
import requests


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
    file_path = "./data/images/cat.jpg"

    # 构造请求数据
    files = {"file": open(file_path, "rb")}

    # 发送 POST 请求
    response = requests.post("http://localhost:8000/classification/predict", files=files)

    # 解析响应
    if response.status_code == 200:
        print("Prediction:", response.text)
    else:
        print("Error:", response.text)
