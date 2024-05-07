import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
from PIL import Image
import numpy as np 
from thop import profile

from .model_zoo import *
from .utils import highlight_text, cvtColor, resize_image, load_class_rgb
from .fastsegformer.fast_segformer import FastSegFormer

class segmentModel:
    """
    支持的模型列表:
    FastSegFormer（脐橙缺陷检测模型）: FastSegFormer-P12, FastSegFormer-E-S0
    """
    def __init__(self, modelName="FastSegFormer-P12") -> None:
        # 检查是否支持模型
        assert modelName in segment_weight_zoo, "modelName should be in {}".format(segment_weight_zoo.keys())
        modelWeightPath = segment_weight_zoo[modelName]
        # 检查本地是否下载模型
        assert os.path.exists(modelWeightPath), "Model weight file: {} ".format(modelWeightPath) + \
                                                "does not exist.\n Please download {} ".format(modelName) + \
                                                "model in " + \
                                                highlight_text("'{}'".format(segment_model_url_zoo[modelName]))
        
        # 加载模型
        if modelName == "FastSegFormer-P12":
            self.model = FastSegFormer(num_classes=3 + 1, backbone="poolformer_s12", Pyramid="multiscale", cnn_branch=True)
        elif modelName == "FastSegFormer-E-S0":
            self.model = FastSegFormer(num_classes=3 + 1, backbone="efficientformerV2_s0", Pyramid="multiscale", cnn_branch=True)
        else:
            raise ValueError("暂不支持该模型")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using {} device!".format(self.device))
        self.model.to(self.device) 
        # 加载权重
        self.model.load_state_dict(torch.load(modelWeightPath, map_location=self.device)) 
        # 推理模式
        self.model.eval()
        self.model_name = modelName

    def preprocess_image(self, imagePath):
        assert os.path.exists(imagePath), "Image file : '{}' does not exist.".format(imagePath)
        image = Image.open(imagePath)
        # 对灰度图像的处理
        image = cvtColor(image)
        self.old_image = copy.deepcopy(image)
        # 对图片进行resize, 默认使用224x224的图片
        image_data, self.nw, self.nh = resize_image(image, size=(224, 224))
        # 1. /255  2. W, H, C -> C, W, H  3. 添加batch
        image_data = np.expand_dims(np.transpose(np.array(image_data, np.float32) / 255.0, (2, 0, 1)), 0)
        # numpy -> torch
        image_tensor = torch.from_numpy(image_data).to(self.device)

        return image_tensor

    def inference(self, image_tensor):
        with torch.no_grad():
            outputs = self.model(image_tensor)

        return outputs

    def postprocess(self, results, classNameFile="./data/classname/orange_defect.names", legend=False, save=False):
        
        # 加载类别和颜色
        colors, classname = load_class_rgb(classNameFile)
        # 后处理
        pr = results[0]
        pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
        pr = pr[int((224 - self.nh) // 2) : int((224 - self.nh) // 2 + self.nh), \
                int((224 - self.nw) // 2) : int((224 - self.nw) // 2 + self.nw)]
        pr = cv2.resize(pr, (self.old_image.size[0], self.old_image.size[1]), interpolation = cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)
        seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [self.old_image.size[0], self.old_image.size[1], -1])
        image = Image.fromarray(np.uint8(seg_img))

        if legend:
            # 在图片右侧新增空白增加图例
            legend_width = int(self.old_image.size[0] * 0.2) # 图例的宽度为原始图像宽度的20%
            legend_height = self.old_image.size[1] # 图例的高度与原始图像的高度相同
            legend_img = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255 # 白色背景
            for i, (name, color) in enumerate(zip(classname, colors)):
                legend_color_block = np.ones((int(legend_height//(len(classname)+1) // 3), int(legend_width//3), 3), dtype=np.uint8) * color # 创建图例颜色块
                legend_img[int(legend_height//(len(classname)+1)*(i+1)):int(legend_height//(len(classname)+1)*(i+1)) + int(legend_height//(len(classname)+1) // 3), int(legend_width//3):int(legend_width//3)*2] = legend_color_block
                cv2.putText(legend_img, name, (int(legend_width//3), int(legend_height//(len(classname)+1)*(i+1)) + int(legend_height//(len(classname)+1)*0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            # 将图例合并到原始图像的右侧
            image = np.hstack((image, legend_img))
            image = Image.fromarray(image)
            if save:
                image.save("detect_result.jpg")
                print("检测结果图片已经保存在当前目录下!")

            return image

        image = Image.blend(self.old_image, image, 0.7)

        if save:
            image.save("detect_result.jpg")
            print("检测结果图片已经保存在当前目录下!")
        
        return image

    def predict(self, imagePath, classNameFile="./data/classname/orange_defect.names", legend=False, save=False):
        image_tensor = self.preprocess_image(imagePath=imagePath)
        outputs = self.inference(image_tensor=image_tensor)
        image = self.postprocess(results=outputs, classNameFile=classNameFile, legend=legend, save=save)

        return image
        
    
    
    def count_params_flops(self, input, view=True):
        flops, params = profile(self.model, inputs=(input.to(self.device),))
        if view:
            print(f"{self.model_name}模型的FLOPS为：{flops / 1e9:.2f} GFLOPS")
            print(f"{self.model_name}模型的参数数量为：{params / 1e6:.2f} M")
        return round(flops/1e9, 2), round(params/1e6, 2)
        
