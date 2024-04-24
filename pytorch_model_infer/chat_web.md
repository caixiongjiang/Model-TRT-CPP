### Text Generation Web UI

#### 快速开始

* 克隆仓库

```bash
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
```

* Linux系统运行`start_linux.sh`，Windows系统上运行`start_windows.bat`，在MacOS系统上运行`start_macos.sh`，或者在Windows子系统Linux（WSL）上运行`start_wsl.bat`。

* 安装环境依赖，可以使用anaconda或者docker，先安装相应的pytorch版本，然后安装相应的依赖项：

```bash
conda create -n textgen python=3.11
conda activate textgen
pip3 install torch torchvision torchaudio # 具体根据官网寻找对应CUDA版本的下载命令
```

* 根据系统选择安装依赖项：
```bash
pip3 install -r requirements_apple_silicon.txt
```

#### 模型准备

下载相应的模型放入如下的位置，以Qwen1.5-7B-Chat的模型为例子：
```
text-generation-webui
├── models
│   ├── Qwen1.5-7B-Chat
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model-00001-of-00004.safetensor
│   │   ├── model-00002-of-00004.safetensor
│   │   ├── model-00003-of-00004.safetensor
│   │   ├── model-00004-of-00004.safetensor
│   │   ├── model.safetensor.index.json
│   │   ├── merges.txt
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
```

#### 启动网页

```bash
python3 server.py
```

点击进入给出的网页。

 