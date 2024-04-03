# Model-TRT-CPP

## Environment

### Pytorch Infer

* 使用python代码进行Pytorch的推理环境：
```bash
cd pytorch_model_infer
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
* 🐳docker镜像：
```bash
```

### TensorRT Infer

* 🐳docker镜像：
```bash

```
## Usage

### Benchmark推理测试

```bash
cd pytorch_model_infer
CUDA_VISIBLE_DEVICES={gpu number} python3 benchmark.py
```

### 图片推理 && 文字推理

```bash
cd pytorch_model_infer
CUDA_VISIBLE_DEVICES={gpu number} python3 predict.py
```

### API接口推理


## Result

| 状态 | 模型 | 论文 | 分辨率 | Pytorch(FPS) | TensorRT(FPS) |
| :-----: | :-----:| :-----: | :------: | :------: | :------: | 
| &#x2610; | ResNet50 | | $224\times 224$ |  | |
| &#x2610; | MobileNet | | |  | |


