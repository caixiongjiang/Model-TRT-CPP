import numpy as np
import torch
from torch.backends import cudnn
import tqdm
cudnn.benchmark = True
import argparse

from nets.classification.model_infer import classifyModel


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





if __name__ == '__main__':
     # 创建一个ArgumentParser对象
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加参数
    parser.add_argument('--modelName', type=str, default='resnet50', help='Model Name')
    parser.add_argument('--repetitions', type=int, default=300, help='Number of repetitions')

    # 解析参数
    args = parser.parse_args()

    # 使用命令行参数创建模型
    model = classifyModel(modelName=args.modelName)

    # 创建一个随机输入
    input = torch.randn(1, 3, 224, 224)

    # benchmark测试
    benchmark(model, input, model.device, repetitions=args.repetitions)
