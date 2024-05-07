import os
import torch
from PIL import Image
import numpy as np
import colorsys
import copy
from thop import profile

from .yolov5.yolov5 import Yolov5
from .yolov8.yolov8 import Yolov8
from .model_zoo import *
from .utils import *
from .utils_bbox import DecodeBox, YOLOv8_DecodeBox

class YOLO_Inference():
    """
    支持模型列表：
    yolov5n yolov5s yolov5m yolov5l yolov5x
    yolov8n yolov8s yolov8m yolov8l yolov8x
    """

    def __init__(self, modelName="yolov5s", 
                 anchors_path="./data/classname/yolo_anchors.txt",
                 anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]) -> None:
        # 检查是否支持模型
        assert modelName in detect_weight_zoo, "modelName should be in {}".format(detect_weight_zoo.keys())
        modelWeightPath = detect_weight_zoo[modelName]
        # 检查本地是否下载模型
        assert os.path.exists(modelWeightPath), "Model weight file: {} ".format(modelWeightPath) + \
                                                "does not exist.\n Please download {} ".format(modelName) + \
                                                "model in " + \
                                                highlight_text("'{}'".format(detect_model_url_zoo[modelName]))
        # 读取anchors配置
        assert os.path.exists(anchors_path), "Anchors file: {} does not exist!".format(anchors_path)
        anchors, anchors_num = get_anchors(anchors_path)
        # 读取
        phi = modelName[-1]
        # 加载模型
        if "yolov5" in modelName:
            self.model = Yolov5(anchors_mask, num_classes=80, phi=phi)
        elif "yolov8" in modelName:
            self.model = Yolov8(num_classes=80, phi=phi)
        else:
            raise ValueError("暂不支持 {} 模型".format(modelName))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using {} device!".format(self.device))
        self.model.to(self.device) 
        # 加载权重
        self.model.load_state_dict(torch.load(modelWeightPath, map_location=self.device))
        self.model.eval()
        self.model_name = modelName
        # 解析类别文件
        self.class_names, self.num_classes = get_classes("./data/classname/coco.names")
        # 加载后处理设置 默认使用coco 80类，输入为640, 640, 默认的类别数量为80
        hsv_tuples = [(x / 80, 1., 1.) for x in range(80)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        if "yolov5" in modelName:
            self.bbox_util = DecodeBox(anchors=anchors, num_classes=self.num_classes, input_shape=(640, 640), anchors_mask=anchors_mask)
        elif "yolov8" in modelName:
            self.bbox_util = YOLOv8_DecodeBox(num_classes=self.num_classes, input_shape=(640, 640))
        self.image_shape = None
        self.letter_box = None
        self.old_image = None

    def preprocess(self, imagePath, letter_box=True):
        assert os.path.exists(imagePath), "Image file : '{}' does not exist.".format(imagePath)
        image = Image.open(imagePath)
        # 计算输入图片的宽高
        self.image_shape = np.array(np.shape(image)[0:2])
        # 是否使用letter_box
        self.letter_box = letter_box
        # 对灰度图像的处理
        image = cvtColor(image)
        self.old_image = copy.deepcopy(image)
        # 对图片进行resize, 默认使用640x640的图片
        image_data = resize_image(image, (640, 640), letter_box)
        # 1. /255  2. W, H, C -> C, W, H  3. 添加batch
        image_data = np.expand_dims(np.transpose(np.array(image_data, dtype='float32') / 255.0, (2, 0, 1)), 0)
        # numpy -> torch
        image_tensor = torch.from_numpy(image_data).to(self.device)

        return image_tensor
    
    def inference(self, image_tensor):
        with torch.no_grad():
            outputs = self.model(image_tensor)

        return outputs

    def postprocess(self, results, conf=0.5, iou=0.5, save=False):
        # 检测头解码
        results = self.bbox_util.decode_box(results)
        # 预测框堆叠并进行非极大抑制
        if "yolov5" in self.model_name:
            results = self.bbox_util.non_max_suppression(torch.cat(results, 1), self.num_classes, (640, 640), 
                                                        self.image_shape, self.letter_box, conf, iou)
            top_label   = np.array(results[0][:, 6], dtype='int32')
        elif "yolov8" in self.model_name:
            results = self.bbox_util.non_max_suppression(results, self.num_classes, (640, 640), 
                        self.image_shape, self.letter_box, conf, iou)
            top_label   = np.array(results[0][:, 5], dtype='int32')
        if results[0] is None:
            results = self.old_image

        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]

        # 图像绘制
        image = draw_boxes(top_label=top_label, class_names=self.class_names,
                           top_boxes=top_boxes, top_conf=top_conf,
                           image=self.old_image, colors=self.colors)
        # 图像保存
        if save:
            image.save("detect_result.jpg")
            print("检测结果图片已经保存在当前目录下!")
        
        return image, top_label, top_conf, top_boxes

    def predict(self, imagePath, letter_box=True, conf=0.5, iou=0.5, save=False):
        image_tensor = self.preprocess(imagePath=imagePath, letter_box=letter_box)
        outputs = self.inference(image_tensor)
        image, _, _, _ = self.postprocess(results=outputs, conf=conf, iou=iou, save=save)

        return image
    
    def count_params_flops(self, input, view=True):
        flops, params = profile(self.model, inputs=(input.to(self.device),))
        if view:
            print(f"{self.model_name}模型的FLOPS为：{flops / 1e9:.2f} GFLOPS")
            print(f"{self.model_name}模型的参数数量为：{params / 1e6:.2f} M")
        return round(flops/1e9, 2), round(params/1e6, 2)
    
    def gen_predict_results(self, top_label, top_conf, top_boxes):
        res = {
            "results": []
        }

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = int(max(1, np.floor(top)))
            left    = int(max(1, np.floor(left)))
            bottom  = int(min(self.old_image.size[1] - 1, np.floor(bottom)))
            right   = int(min(self.old_image.size[0] - 1, np.floor(right)))

            box_p = [left, top, right, bottom] # 左上右下 

            temp_result = {
                "class": predicted_class,
                "score": round(score.astype('float'), 2),
                "box": box_p
            }
            res["results"].append(temp_result)
        
        return res