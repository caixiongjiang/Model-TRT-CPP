import timm 
import requests
import uuid 


headers = {
    "appId": "classification_infer",
    "requestId": str(uuid.uuid4())
}


if __name__ == "__main__":
    # model = timm.create_model('vit_tiny_r_s16_p8_224.augreg_in21k_ft_in1k')
    # print(model.default_cfg)
    # model = timm.create_model('vit_small_r26_s32_224.augreg_in21k_ft_in1k')
    # print(model.default_cfg)
    # model = timm.create_model('vit_large_r50_s32_224.augreg_in21k_ft_in1k')
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
