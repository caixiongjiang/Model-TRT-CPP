# Model-TRT-CPP

## 环境配置

### Python 推理

* 使用python代码进行**Pytorch**和**TensorRT**的推理环境：
```bash
cd pytorch_model_infer
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
* 🐳docker镜像,通过dockerfile构建：
```bash
cd pytorch_model_infer/docker
docker build -t ${镜像名} .
```

### C++ 推理

* 🐳docker镜像：
```bash

```
## 使用

### Benchmark推理测试

```bash
cd pytorch_model_infer
CUDA_VISIBLE_DEVICES={gpu number} python3 benchmark.py
```

### 图片推理

```bash
# 分类，检测，分割
cd pytorch_model_infer
CUDA_VISIBLE_DEVICES={gpu number} python3 predict.py
```

### API推理接口

```bash
# 相关参数在config/config.yaml里设置
# 分类，检测，分割
cd pytorch_model_infer
CUDA_VISIBLE_DEVICES={gpu number} python3 api.py
```
### LLM

#### Chat API接口

```bash
cd pytorch_model_infer
CUDA_VISIBLE_DEVICES={gpu number} python3 chat_api.py
```

#### Chat OpenAI格式 API

具体见[openai.md](./pytorch_model_infer/openai.md)

#### Chat Web

```bash
cd pytorch_model_infer
CUDA_VISIBLE_DEVICES={gpu number} streamlit run chat_web.py --server.address 0.0.0.0 --server.port {your port}
```




## Result
> 本仓库的所有测试均在单卡4090D上进行。

* 图像模型：

| 模型 | 论文/仓库 | 测试模型 | 分辨率 | Pytorch(FPS) | TensorRT(FPS) |
| :-----:| :-----: | :------: | :------: | :------: | :------: | 
| ResNet系列 | [论文地址](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) | ResNet50 | $224\times 224$ |  | |
| MobileNet系列 | |  | $224\times 224$ |  |  |
| Vision Transformer系列 | [论文地址](https://arxiv.org/pdf/2010.11929.pdf) | ViT-T | $224\times 224$ |  |  |
| YOLOv5系列 | [仓库地址](https://github.com/ultralytics/yolov5) | yolov5s | $640\times 640$ |  |  |
| YOLOv8系列 | [仓库地址](https://github.com/ultralytics/ultralytics) | yolov8m | $640\times 640$ |  |  |

* LLM: 

| 模型 | 论文/仓库 | 测试模型 | 吞吐量 | token/s |
| :-----:| :-----: | :------: | :------: | :------: |
| Qwen1.5 | [仓库地址](https://github.com/QwenLM/Qwen1.5) | Qwen1.5-7B-Chat |  |  |
| ChatGLM3 | [仓库地址](https://github.com/THUDM/ChatGLM3) | ChatGLM3-6B |  |  |



## 计划

* 图像模型:

| 模型 | benchmark | 图像预测 | api接口 | Python TensorRT推理 | C++ TensorRT推理 |
| :-----: | :-----: | :-----: | :------: | :------: | :------: |
| ResNet系列 | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |  |
| MobileNet系列 |  | | | | |
| ViT系列 | :white_check_mark: | :white_check_mark: | :white_check_mark: | | |
| YOLOv5系列 | :white_check_mark: | :white_check_mark: | :white_check_mark: | | |
| YOLOv8系列 | :white_check_mark: | :white_check_mark: | :white_check_mark: | | |

* LLM:

| 模型 | 单轮对话api | web多轮对话 | 单轮对话流式输出 | vLLM并发 |
| :-----: | :-----: | :-----: | :------: | :------: |
| ChatGLM3系列 | :white_check_mark: | | :white_check_mark: | :white_check_mark: |
| Qwen1.5系列 | :white_check_mark: | | :white_check_mark: | :white_check_mark: |



