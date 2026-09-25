# SmallNeko

SmallNeko 是一个按阶段学习和实验小型语言模型的项目。

当前阶段：

- `stages/stage0/`：最小训练管线和字符级 tokenizer 调试；
- `stages/stage1/`：标准 decoder-only Transformer 自回归基线；
- `stages/stage2/`：后续用于受控推理评估集。

阶段一的运行说明在 [stages/stage1/README.md](stages/stage1/README.md)，数据格式说明在 [stages/stage1/data/README.md](stages/stage1/data/README.md)。
