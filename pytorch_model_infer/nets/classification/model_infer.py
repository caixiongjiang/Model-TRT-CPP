import torch
import torch.nn as nn

import timm
from thop import profile
from torchvision import transforms

from PIL import Image

import os

from .model_zoo import *
from .utils import *
 
# TODO: 增加其他分类模型支持并测试
class classifyModel:
    """
    支持的模型列表:
    resnet系列: resnet18, resnet34, resnet50, resnet101, resnet152
    vit系列: vit-t, vit-s, vit-l
    """
    def __init__(self, modelName="resnet50") -> None:
        # 检查是否支持模型
        assert modelName in class_model_zoo, "modelName should be in {}".format(class_model_zoo.keys())
        assert modelName in class_weight_zoo, "modelName should be in {}".format(class_weight_zoo.keys())
        modelWeightPath = class_weight_zoo[modelName]
        # 检查本地是否下载模型
        assert os.path.exists(modelWeightPath), "Model weight file: {}".format(modelWeightPath) + \
                                                " does not exist.\n Please download {}".format(modelName) + \
                                                "model in" + \
                                                highlight_text("'{}'".format(class_model_url_zoo[modelName]))
        # 加载模型
        self.model = timm.create_model(class_model_zoo[modelName])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using {} device!".format(self.device))
        self.model.to(self.device) 
        # 加载权重
        if modelWeightPath.split("/")[-1].split(".")[-1] == "npz":
            # 目前只能更改缓存文件的位置来实现本地加载npz权重
            pretrained_cfg = timm.create_model(class_model_zoo[modelName]).default_cfg
            pretrained_cfg['file'] = modelWeightPath
            self.model = timm.create_model(class_model_zoo[modelName], pretrained=True, pretrained_cfg=pretrained_cfg).to(self.device)
        else:
            self.model.load_state_dict(torch.load(modelWeightPath, map_location=self.device)) 
        # 推理模式
        self.model.eval()
        self.model_name = modelName
    
    def preprocess_image(self, imagePath):
        assert os.path.exists(imagePath), "Image file : '{}' does not exist.".format(imagePath)
        image = Image.open(imagePath)
        # Preprocess the image
        # 分类模型默认的推理输入分辨率为224x224
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
        image = preprocess(image)
        image = torch.unsqueeze(image, dim=0).to(self.device)
        return image
    
    def inference(self, imageTensor):
        with torch.no_grad():
            outputs = self.model(imageTensor)

        return outputs
    
    def postprocess(self, result, classNameFile):
        _, predicted = torch.max(result.cpu(), 1)
        imagenet_classes, _ = get_classes(classNameFile)
        category_name = imagenet_classes[predicted]
        print(f"类别名称: {category_name}")

        return category_name

    def predict(self, imagePath, classNameFile):
        image_tensor = self.preprocess_image(imagePath)
        res = self.postprocess(self.inference(image_tensor), classNameFile)
        
        return res
    
    
    def count_params_flops(self, input, view=True):
        flops, params = profile(self.model, inputs=(input.to(self.device),))
        if view:
            print(f"{self.model_name}模型的FLOPS为：{flops / 1e9:.2f} GFLOPS")
            print(f"{self.model_name}模型的参数数量为：{params / 1e6:.2f} M")
        return round(flops/1e9, 2), round(params/1e6, 2)
        

