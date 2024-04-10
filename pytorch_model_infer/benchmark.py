import numpy as np
import torch
from torch.backends import cudnn
import tqdm
cudnn.benchmark = True
import argparse
import ast

from nets.classification.model_infer import classifyModel
from nets.detection.yolo_infer import YOLO_Inference
from nets.trt_infer import TRTWrapper


def benchmark(model, input, device, repetitions=300):
    # 模型初始化阶段已经进入CUDA设备
    model = model
    input = input.to(device)

    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model.inference(input)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            _ = model.inference(input)
            ender.record()
            torch.cuda.synchronize() # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
            timings[rep] = curr_time

    avg = timings.sum()/repetitions
    fps = 1000.0/avg

    flops, params = model.count_params_flops(input, view=False)
    print('\nmodel: {}'.format(model.model_name))
    print('\nparams={}M'.format(params))
    print('\nflops={}G'.format(flops))
    print('\navg={}'.format(avg))
    print('\nfps={}'.format(fps))


def trt_benchmark(engine, input, repetitions=300):

    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    for _ in range(100):
        _ = engine(input)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    # Initialize CUDA events for measuring time
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Initialize an array to store timings
    timings = np.zeros((repetitions, 1))
    
    print('Testing TensorRT inference...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            _ = engine(input)
            ender.record()
            torch.cuda.synchronize()  # Wait for GPU tasks to complete
            curr_time = starter.elapsed_time(ender)  # Time elapsed from starter to ender, in milliseconds
            timings[rep] = curr_time

    avg = timings.sum() / repetitions
    fps = 1000.0 / avg
    
    print('\nTensorRT engine: {}'.format(engine.trt_name))
    print('\nAvg: {}'.format(avg))
    print('\nFPS: {}'.format(fps))



def to_tensor(s):
    try:
        shape = ast.literal_eval(s)
        if isinstance(shape, tuple) and len(shape) == 4:
            return shape
        else:
            raise argparse.ArgumentTypeError("输入的形状应为一个长度为4的元组")
    except ValueError:
        raise argparse.ArgumentTypeError("无法解析输入")



if __name__ == '__main__':
     # 创建一个ArgumentParser对象
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加参数
    parser.add_argument('--modelName', type=str, default='resnet50', help='Model Name')
    parser.add_argument('--mode', type=str, default='torch', choices=['torch', 'trt'], help='Inference mode')
    parser.add_argument('--input', type=to_tensor, default=(1, 3, 224, 224), help='输入一个张量形状，格式为 "(1, 3, 224, 224)"')
    parser.add_argument('--type', type=str, default='classify', choices=['classify', 'detect', 'segment'], help='推理的模型类型')
    parser.add_argument('--repetitions', type=int, default=300, help='Number of repetitions')

    # 解析参数
    args = parser.parse_args()

    if args.mode == "torch":
        # 使用pytorch模型推理
        if args.type == "classify":
            model = classifyModel(modelName=args.modelName)
        elif args.type == "detect":
            model = YOLO_Inference(modelName=args.modelName)
        elif args.type == "segment":
            pass
        else:
            raise ValueError("暂不支持当前的任务相关的模型")
        # 创建一个随机输入
        input = torch.randn(args.input)
        # benchmark测试
        benchmark(model, input, model.device, repetitions=args.repetitions)

    elif args.mode == "trt":
        # 使用tensor文件推理
        model_name = args.modelName
        if args.type == "classify":
            engine = TRTWrapper(f"./data/TRT/classification/{model_name}.trt", ["output"])
        elif args.type == "detect":
            pass
        elif args.type == "segment":
            pass
        else:
            raise ValueError("暂不支持当前的任务相关的模型")
        # 创建一个随机输入
        input = torch.randn(args.input).cuda()
        trt_benchmark(engine, dict(input=input), repetitions=300)
    else:
        # 暂时不支持该推理模式
        raise ValueError("抱歉，暂时不支持该推理模式，请选择其他支持的推理模式。")
