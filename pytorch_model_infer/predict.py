import argparse
import torch
from nets.classification.model_infer import classifyModel


if __name__ == '__main__':
    # 创建一个ArgumentParser对象
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加参数
    parser.add_argument('--modelName', type=str, default='resnet50', help='Model Name')
    parser.add_argument('--imagePath', type=str, default='./data/images/cat.jpg', help='The path to the image to predict.')
    parser.add_argument('--classNameFile', type=str, default='./data/classname/imagenet1k.names', help='The path to the file containing class names.')

    # 解析参数
    args = parser.parse_args()

    # 使用命令行参数创建模型
    model = classifyModel(modelName=args.modelName)

    # 使用命令行参数进行预测
    model.predict(args.imagePath, args.classNameFile)

    # 打印模型参数数量, 计算量
    model.count_params_flops(torch.randn(1, 3, 224, 224))
    
