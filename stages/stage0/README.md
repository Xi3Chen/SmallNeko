# 阶段 0：训练管线与对话数据调试

阶段 0 的目标是验证最小训练流程能够正常工作：数据读取、tokenizer、前向传播、反向传播、验证、checkpoint 保存与恢复，以及基本文本生成。

本阶段不把这批数据当作知识底座，也不据此判断模型是否具备跨领域推理能力。当前数据主要是猫娘风格的单轮用户问题与助手回答，适合检查训练管线和对话格式。

## 当前实现

`src/char_tokenizer.py` 提供阶段 0 的 Unicode 字符级 tokenizer。它只使用训练集建立字符表，并将验证集中新出现的字符映射到 `<unk>`。`<|im_start|>` 和 `<|im_end|>` 等对话边界会作为独立 token 保存。

在 WSL 中运行：

```bash
cd /mnt/k/MyLm/SmallNeko/stages/stage0
source ../../.venv/bin/activate
python -m pip install -r ../../requirements.txt

cd src
python build_tokenizer.py
python prepare_data.py
python -m unittest discover -s . -p 'test_*.py'
```

生成的 tokenizer 文件位于 `data/processed/char_tokenizer.json`。它只用于阶段 0 调试，不作为后续正式预训练 tokenizer。所有阶段共用项目根目录的 `.venv`，每个阶段仍然可以通过自己的 `requirements.txt` 声明额外依赖。

安装阶段 0 依赖后，使用默认的小模型配置开始训练：

```bash
python train.py --max-iters 1000
```

训练前建议先用很小的配置确认可以过拟合：

```bash
python train.py --batch-size 4 --block-size 128 --n-layer 2 --n-head 2 --n-embd 128 --max-iters 200
```

检查点和训练配置会保存到 `experiments/baseline/`。

## 查看生成效果

训练完成后，在 WSL 中运行交互模式：

```bash
cd /mnt/k/MyLm/SmallNeko/stages/stage0/src
source ../../../.venv/bin/activate
python generate.py --checkpoint ../experiments/baseline/best.pt --device cuda
```

输入问题后按回车即可生成回答，输入空行退出。也可以直接测试一个固定问题：

```bash
python generate.py --checkpoint ../experiments/baseline/best.pt --device cuda \
  --prompt "请介绍一下你自己。" --max-new-tokens 200 --temperature 0.8
```

`--temperature` 越低越稳定，`--top-k` 控制候选字符数量。阶段 0 使用字符级 tokenizer，生成速度和文本质量都有限，这是管线验证结果，不代表后续模型能力。

## 数据

原始数据位于 `data/raw/`：

- `chat_data.json`：原始 JSON 记录，来自旧项目的源数据；
- `train.txt`：旧项目已经划分好的训练文本；
- `val.txt`：旧项目已经划分好的验证文本。

当前不复制旧项目的 `train.bin`、`val.bin` 和 `meta.pkl`。这些文件是旧 nanoGPT 的字符级编码产物，依赖旧 tokenizer，不适合作为 SmallNeko 的输入格式。

## 使用约束

- 阶段 0 可以使用当前对话数据验证端到端训练；
- 数据清洗和 tokenizer 应由 SmallNeko 自己完成；
- 保留原始文件，不在 `data/raw/` 中直接覆盖数据；
- 清洗后的文件放入 `data/processed/`；
- 模型、配置和运行结果分别放入 `src/`、`configs/` 和 `experiments/`；
- 训练前删除空回答、乱码样本，并记录清洗数量；
- 训练前检查训练集和验证集的问题是否重复。

## 阶段 0 完成标准

- 能读取原始对话并构造训练样本；
- 能在小样本上过拟合，证明 loss 和反向传播正常；
- 能使用独立验证集计算 loss；
- 能保存并恢复 checkpoint；
- 能用固定提示词生成文本；
- 能记录 tokenizer、参数量、数据版本和训练配置。
