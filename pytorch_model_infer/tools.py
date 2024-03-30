import torch
import onnx
import onnxsim
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

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


if __name__ == "__main__":
    pt2onnx("resnet50", classify=True, simplify=True)
    onnx_inference("resnet50", np.random.randn(1, 3, 224, 224).astype(np.float32), classify=True, simplify=True)
    onnx_predict("resnet50", "./data/images/cat.jpg",classify=True, simplify=True)



