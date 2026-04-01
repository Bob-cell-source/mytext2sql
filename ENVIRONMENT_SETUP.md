# 项目环境配置清单

这份文档说明当前项目的推荐运行环境、依赖安装方式、环境变量配置，以及使用 LLaMA-Factory 进行 SFT 训练时的额外要求。

## 1. 环境划分

当前项目建议分成两套环境理解：

- 本仓库运行环境
  用于：
  - SQL 执行
  - SFT 数据合成
  - 数据清洗与格式转换

- LLaMA-Factory 训练环境
  用于：
  - 单轮 SFT
  - 整体轨迹版 SFT

不建议把所有依赖都混到一个最小 `requirements.txt` 里，尤其是：

- `torch`
- `flash-attn`
- `bitsandbytes`
- `deepspeed`

这些依赖和 CUDA、驱动、显卡环境强绑定。

## 2. Python 版本建议

推荐：

- Python `3.10` 到 `3.12`

如果你主要使用：

- LLaMA-Factory
- Transformers
- LoRA / bf16 训练

建议优先使用：

- Python `3.10` 或 `3.11`

## 3. 本仓库基础依赖

文件：

- [requirements.txt](/root/text2sql_RL/requirements.txt)

包含：

- `numpy`
- `openai`
- `PyMySQL`

安装方式：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

这套依赖足够支持：

- [sql_exe.py](/root/text2sql_RL/sql_exe.py)
- [final_code.py](/root/text2sql_RL/final_code.py)
- [scripts/synthesize_sft_trajectories.py](/root/text2sql_RL/scripts/synthesize_sft_trajectories.py)
- [scripts/prepare_llamafactory_sft.py](/root/text2sql_RL/scripts/prepare_llamafactory_sft.py)

## 4. 训练相关 Python 依赖

文件：

- [requirements-train.txt](/root/text2sql_RL/requirements-train.txt)

包含：

- `accelerate`
- `datasets`
- `peft`
- `sentencepiece`
- `transformers`
- `trl`

安装方式：

```bash
pip install -r requirements-train.txt
```

说明：

- 这里只放和训练脚本、数据处理、Hugging Face 生态直接相关的通用包
- `torch` 没有写进这个文件，因为它需要根据你的 CUDA 版本单独安装

## 5. PyTorch 安装建议

由于你的机器是 NVIDIA GPU，推荐根据实际 CUDA 环境单独安装 PyTorch。

安装前先确认：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

如果你还没有装 PyTorch，建议按 PyTorch 官方安装页选择对应命令：

- https://pytorch.org/get-started/locally/

示例：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

注意：

- 上面的 `cu124` 只是示例
- 你应当根据本机 CUDA / 驱动情况选择实际版本

## 6. LLaMA-Factory 安装建议

如果你使用 LLaMA-Factory，建议按官方方式安装，而不是只靠一个 requirements 文件。

官方文档：

- 安装文档（英文）：
  https://llamafactory.readthedocs.io/en/latest/getting_started/installation.html
- 安装文档（中文）：
  https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/installation.html

官方安装方式摘要：

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
pip install -r requirements/metrics.txt
```

根据官方文档，如果出现依赖冲突，也可以尝试：

```bash
pip install --no-deps -e .
```

安装完成后验证：

```bash
llamafactory-cli version
```

说明：

- 这是官方当前推荐方式
- 我没有把 `llamafactory` 直接写进本仓库的 `requirements-train.txt`
- 原因是它本身还带有额外依赖分组和训练环境约束，更适合独立安装

## 7. 数据合成所需环境变量

本仓库的数据合成和 SQL 执行依赖以下环境变量。

### 7.1 Teacher 模型

```bash
export DASHSCOPE_API_KEY="你的key"
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LLM_MODEL="qwen3-max"
```

当前代码读取位置：

- [agent_rl/teacher_rollout.py](/root/text2sql_RL/agent_rl/teacher_rollout.py)

### 7.2 StarRocks / MySQL 执行环境

```bash
export DB_HOST="127.0.0.1"
export DB_PORT="9030"
export DB_USER="root"
export DB_PASSWORD=""
export DB_NAME="TGAC"
```

当前代码读取位置：

- [agent_rl/sql_env.py](/root/text2sql_RL/agent_rl/sql_env.py)
- [final_code.py](/root/text2sql_RL/final_code.py)

## 8. 一套推荐的最小可用环境

如果你只想把当前仓库跑起来，建议步骤如下：

### 8.1 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 8.2 安装基础依赖

```bash
pip install -r requirements.txt
```

### 8.3 配置环境变量

```bash
export DASHSCOPE_API_KEY="你的key"
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LLM_MODEL="qwen3-max"

export DB_HOST="127.0.0.1"
export DB_PORT="9030"
export DB_USER="root"
export DB_PASSWORD=""
export DB_NAME="TGAC"
```

### 8.4 运行数据合成

```bash
python3 scripts/synthesize_sft_trajectories.py \
  --attempts 16 \
  --max-turns 3 \
  --keep-top-k 4 \
  --workers 4 \
  --outdir output/sft_synthesis_full_v2
```

### 8.5 处理 SFT 数据

```bash
python3 scripts/prepare_llamafactory_sft.py \
  --input output/sft_synthesis_full_v2/sft_multiturn_action_focused.jsonl \
  --full-input output/sft_synthesis_full_v2/sft_multiturn_full.jsonl \
  --outdir output/llamafactory_sft_v5
```

## 9. 一套推荐的训练环境

如果你接下来要做 SFT，建议：

1. 先在当前虚拟环境里安装 PyTorch
2. 再安装 [requirements-train.txt](/root/text2sql_RL/requirements-train.txt)
3. 再独立安装 LLaMA-Factory

示例顺序：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-train.txt

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
pip install -r requirements/metrics.txt
```

## 10. 当前项目文件与依赖对应关系

### 数据合成

- [scripts/synthesize_sft_trajectories.py](/root/text2sql_RL/scripts/synthesize_sft_trajectories.py)
- [agent_rl/teacher_rollout.py](/root/text2sql_RL/agent_rl/teacher_rollout.py)
- [agent_rl/sql_env.py](/root/text2sql_RL/agent_rl/sql_env.py)

核心依赖：

- `openai`
- `PyMySQL`
- `numpy`

### 数据处理

- [scripts/prepare_llamafactory_sft.py](/root/text2sql_RL/scripts/prepare_llamafactory_sft.py)

核心依赖：

- 标准库为主

### 训练

- [training/llamafactory/sft_action_focused_lora.yaml](/root/text2sql_RL/training/llamafactory/sft_action_focused_lora.yaml)
- [training/llamafactory/sft_full_trajectory_lora.yaml](/root/text2sql_RL/training/llamafactory/sft_full_trajectory_lora.yaml)

核心依赖：

- `torch`
- `transformers`
- `datasets`
- `peft`
- `accelerate`
- `trl`
- `llamafactory`

## 11. 备注

- 如果你以后想把训练依赖也完全纳入本仓库，可再补一个 `environment.yml` 或 Dockerfile
- 当前给出的方式更适合你先把实验快速跑起来
