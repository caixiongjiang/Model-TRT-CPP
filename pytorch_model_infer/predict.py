import argparse
import torch
from nets.classification.model_infer import classifyModel
from nets.detection.yolo_infer import YOLO_Inference


if __name__ == '__main__':
    # 创建一个ArgumentParser对象
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加参数
    parser.add_argument('--modelName', type=str, default='resnet50', help='Model Name')
    parser.add_argument('--imagePath', type=str, default='./data/images/cat.jpg', help='The path to the image to predict.')
    parser.add_argument('--classNameFile', type=str, default='./data/classname/imagenet1k.names', help='The path to the file containing class names.')
    parser.add_argument('--type', type=str, default='classify', choices=['classify', 'detect', 'segment'], help='推理的模型类型')

    # 解析参数
    args = parser.parse_args()

    # 使用命令行参数进行预测
    if args.type == "classify":
        model = classifyModel(modelName=args.modelName)
        model.predict(args.imagePath, args.classNameFile)
        model.count_params_flops(torch.randn(1, 3, 224, 224))
    elif args.type == "detect":
        model = YOLO_Inference(modelName=args.modelName)
        model.predict(args.imagePath, save=True)
        model.count_params_flops(torch.randn(1, 3, 640, 640))
    elif args.type == "segment":
        pass
    else:
        raise ValueError("暂不支持当前的任务相关的模型")
