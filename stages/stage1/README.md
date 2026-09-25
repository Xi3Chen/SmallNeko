# 阶段 1：标准 Transformer 自回归基线

本阶段建立后续 Mamba、Looped Transformer 和注意力残差实验共同使用的标准 decoder-only Transformer 基线。模型采用 pre-LN、因果自注意力、MLP 和权重绑定的语言模型头，默认配置约 0.1B 参数。

## 目录

```text
stage1/
├── data/
│   ├── raw/
│   └── processed/
├── configs/
├── experiments/
└── src/
```

所有命令在 WSL 中执行：

```bash
cd /mnt/k/MyLm/SmallNeko/stages/stage1
source ../../.venv/bin/activate
python -m pip install -r ../../requirements.txt
```

## 准备数据

将 UTF-8 的 `train.txt` 和 `val.txt` 放入 `data/raw/`。如果手头是问答 JSON，可以运行：

也可以按照 `configs/data-mixture.json` 的配比，从多个中文 Wikipedia、OpenWebMath 分片和阶段 0 对话中自动抽取约 500M 字节文本：

```bash
python src/download_data.py
```

如果 Dataset Viewer 被限流，使用公开 parquet 分片路径：

```bash
python src/materialize_shards.py
```

该命令会下载配置中的多个分片并在本地按配比抽取，结果和实际数量记录在 `data/raw/download_report.json`。完整分片保存在 `data/downloads/`，由 `.gitignore` 排除。当前扩容配置为中文 Wikipedia 70%、OpenWebMath 28%、阶段 0 对话 2%。

```bash
python src/format_chat.py \
  --input /path/to/chat.json \
  --output-dir data/raw \
  --val-ratio 0.1 \
  --seed 1337
```

如果只是想先复用阶段 0 的旧数据验证阶段一，可以直接复制文本源文件：

```bash
cp ../../stage0/data/raw/train.txt data/raw/train.txt
cp ../../stage0/data/raw/val.txt data/raw/val.txt
```

也可以从阶段 0 的原始 JSON 重新划分，脚本会跳过空记录，并按第一个用户问题去重：

```bash
python src/format_chat.py --input ../../stage0/data/raw/chat_data.json --output-dir data/raw --val-ratio 0.1 --seed 1337
```

然后构建固定字节 tokenizer 并编码数据：

```bash
python src/build_tokenizer.py
python src/prepare_data.py
```

## 测试和训练

```bash
python -m unittest discover -s src -p 'test_*.py'
python src/train.py --device cuda
```

建议先用短配置确认显存和速度：

```bash
python src/train.py \
  --out-dir experiments/smoke \
  --batch-size 4 \
  --block-size 128 \
  --n-layer 2 \
  --n-head 2 \
  --n-embd 128 \
  --dropout 0.0 \
  --max-iters 20 \
  --eval-interval 10 \
  --eval-iters 2
```

默认配置位于 `configs/baseline.json`，并由训练脚本自动读取。它使用约 99.8M 参数、每步 32 x 512 = 16,384 token、30,000 次更新，约采样 4.92 亿训练 token。每个实验目录会保存 `config.json`、`metrics.jsonl`、`best.pt` 和 `last.pt`；`last.pt` 每 500 次更新覆盖保存，包含优化器与随机数状态，可在意外停止后继续训练：

```bash
python src/train.py \
  --resume experiments/baseline-0.1b/last.pt \
  --max-iters 30000
```

## 查看生成

```bash
python src/generate.py \
  --checkpoint experiments/baseline-0.1b/best.pt \
  --device cuda \
  --prompt "请介绍一下你自己。" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 40
```

`--temperature 0` 使用贪心解码；`--top-p` 可以进一步限制采样范围。字节级 tokenizer 的优点是词表固定、Unicode 覆盖完整，代价是中文通常会占用更多 token，因此阶段 1 的上下文长度和生成速度需要单独记录。

要保存固定提示词的生成结果，运行：

```bash
python src/evaluate.py \
  --checkpoint experiments/baseline-0.1b/best.pt \
  --prompts configs/prompts.json \
  --output experiments/baseline/fixed_prompts.jsonl
```

阶段 1 只回答“标准 Transformer 基线能否稳定训练，以及它在统一数据和预算下达到什么水平”。受控推理评估集属于后续阶段。
