import torch
import onnx
import onnxsim
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import tensorrt as trt

from nets.trt_infer import TRTWrapper

# 目前仅支持静态batch大小的模型转换

from nets.classification.model_infer import classifyModel

def pt2onnx(model_name, classify=False, detect=False, segment=False, simplify=False):
    if sum([classify, detect, segment]) != 1:
        raise ValueError("Exactly one of 'classify', 'detect', or 'segment' must be set to True.")

    if classify:
        classification_model = classifyModel(model_name)
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        onnx_file_path = "./data/ONNX/classification/{}.onnx".format(model_name)
    if detect:
        pass
    if segment:
        pass
    # 开始转换模型
    torch.onnx.export(model=classification_model.model, 
                        args=dummy_input,         # 输入的尺寸
                        f=onnx_file_path,         # 输出onnx模型的位置
                        export_params=True,       # 输出模型是否可训练
                        verbose=True,             # 是否打印模型转化信息
                        input_names=["input"],    # 输入节点的名称
                        output_names=["output"],  # 输出节点的名称
                        do_constant_folding=True, # 是否执行常量折叠优化
                        opset_version=11)
    print("ONNX 模型已保存到:", onnx_file_path)

    if simplify:
        # 简化模型
        onnx_sim_file_path = "./data/ONNX/classification/{}_sim.onnx".format(model_name)
        onnx_model = onnx.load(onnx_file_path)
        sim_model, check = onnxsim.simplify(onnx_model)
        if check:
            onnx.save(sim_model, onnx_sim_file_path)
            print("ONNX 简化模型已保存到:", onnx_sim_file_path)
        else:
            print("ONNX 简化模型失败")

def onnx_inference(model_name, input, classify=False, detect=False, segment=False, simplify=False):
    if sum([classify, detect, segment]) != 1:
        raise ValueError("Exactly one of 'classify', 'detect', or 'segment' must be set to True.")
    
    if classify:
        if simplify:
            onnx_file_path = "./data/ONNX/classification/{}_sim.onnx".format(model_name)
        else:
            onnx_file_path = "./data/ONNX/classification/{}.onnx".format(model_name)
    if detect:
        pass
    if segment:
        pass
    
    # 创建一个 ONNX Runtime 的推理会话，并指定使用 GPU
    session = ort.InferenceSession(onnx_file_path, 
                                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # 获取输入输出节点的名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # ONNX Runtime 推理
    pred = session.run([output_name], {input_name: input})[0]
    print("ONNX 推理结果的张量维度:", pred.shape)

    return pred
        

def onnx_predict(model_name, image_path, classify=False, detect=False, segment=False, simplify=False):
    if sum([classify, detect, segment]) != 1:
        raise ValueError("Exactly one of 'classify', 'detect', or 'segment' must be set to True.")

    if classify:
        # 前处理
        image = Image.open(image_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
        image = preprocess(image)
        image = torch.unsqueeze(image, dim=0).numpy()
        # ONNX 推理
        pred = onnx_inference(model_name, image, classify=classify, detect=detect, segment=segment, simplify=simplify)
        # 后处理
        predicted = np.argmax(pred, axis=1)[0]
        with open("./data/classname/imagenet1k.names", "r") as file:
            lines = file.readlines()
        imagenet_classes = [line.strip() for line in lines]
        category_name = imagenet_classes[predicted]
        print(f"类别名称: {category_name}")

    if detect:
        pass
    if segment:
        pass

#  容器内通过Pytorch安装的CUDA和CUDNN版本，CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0 TensorRT=8.6.0
def onnx2trt(model_name, classify=False, detect=False, segment=False, use_simplify=False):
    """
    ONNX 转换为 TensorRT 模型
    engine: TensorRT 模型
    builder: TensorRT 构建器
    config: TensorRT 配置
    parser: 解析ONNX 
    """

    if sum([classify, detect, segment]) != 1:
        raise ValueError("Exactly one of 'classify', 'detect', or 'segment' must be set to True.")
    
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)( 
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 
    network = builder.create_network(EXPLICIT_BATCH) 

    # parse ONNX
    if classify:
        if use_simplify:
            onnx_model = onnx.load(f"./data/ONNX/classification/{model_name}_sim.onnx")
        else:
            onnx_model = onnx.load(f"./data/ONNX/classification/{model_name}.onnx")
    if detect:
        pass
    if segment:
        pass

    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_model.SerializeToString()): 
        error_msgs = '' 
        for error in range(parser.num_errors): 
            error_msgs += f'{parser.get_error(error)}\n' 
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}') 
    
    config = builder.create_builder_config()
    # config.max_workspace_size = 1 << 30 # NVIDIA即将弃用改代码
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    profile = builder.create_optimization_profile()
    # 如果有多 Batch、多输入、动态 shape 的需求，都可以通过多次调用 set_shape 函数进行设置。
    # set_shape 函数接受的传参分别是：输入节点名称，可接受的最小输入尺寸，最优的输入尺寸，可接受的最大输入尺寸。
    # 一般要求这三个尺寸的大小关系为单调递增
    if classify:
        profile.set_shape('input', [1, 3, 224, 224], [1, 3, 224, 224], [1, 3, 224, 224])
    if detect:
        pass
    if segment:
        pass
    config.add_optimization_profile(profile)
    # create engine
    # Note：因为外层运行脚本时会使用CUDA_VISIBLE_DEVICES命令，默认只有0卡，device的index=0
    with torch.cuda.device(0):
        # TODO：build_engine即将弃用
        engine = builder.build_engine(network, config)
    # save engine
    if classify:
        print(f"generating {model_name}.trt file")
        with open(f'./data/TRT/classification/{model_name}.trt', 'wb') as f:
            f.write(bytearray(engine.serialize()))
    if detect:
        pass
    if segment:
        pass
    print(f"generating {model_name}.trt file done!")


def trt_inference(model_name, input, classify=False, detect=False, segment=False):
    if sum([classify, detect, segment]) != 1:
        raise ValueError("Exactly one of 'classify', 'detect', or 'segment' must be set to True.")

    trt_model = TRTWrapper(f"./data/TRT/classification/{model_name}.trt", ["output"])
    if classify:
        expect_shape = (1, 3, 224, 224)
        assert input.shape == expect_shape, "输入的张亮形状应该为{}".format(expect_shape)
    if detect:
        pass
    if segment:
        pass

    pred = trt_model(dict(input=input.cuda())) 
    print("TensorRT 推理结果的张量维度:", pred["output"].shape)
    return pred["output"]


def trt_predict(model_name, image_path, classify=False, detect=False, segment=False):
    if sum([classify, detect, segment]) != 1:
        raise ValueError("Exactly one of 'classify', 'detect', or 'segment' must be set to True.")
    
    if classify:
        # 前处理
        image = Image.open(image_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
        image = preprocess(image)
        image = torch.unsqueeze(image, dim=0)
        # TRT推理
        pred = trt_inference(model_name, image, classify=classify, detect=detect, segment=segment)
        # 后处理
        with open("./data/classname/imagenet1k.names", "r") as file:
            lines = file.readlines()
        _, predicted = torch.max(pred.cpu(), 1)
        imagenet_classes = [line.strip() for line in lines]
        category_name = imagenet_classes[predicted]
        print(f"类别名称: {category_name}")

    if detect:
        pass
    if segment:
        pass


if __name__ == "__main__":
    pt2onnx("resnet50", classify=True, simplify=True)
    onnx_inference("resnet50", np.random.randn(1, 3, 224, 224).astype(np.float32), classify=True, simplify=True)
    onnx_predict("resnet50", "./data/images/cat.jpg", classify=True, simplify=True)
    onnx2trt("resnet50", classify=True, use_simplify=True)
    trt_inference("resnet50", torch.randn(1, 3, 224, 224), classify=True)
    trt_predict("resnet50", "./data/images/cat.jpg", classify=True)



