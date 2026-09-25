# SmallNeko 阶段目录

SmallNeko 按学习阶段隔离代码、数据和实验产物，避免不同阶段的训练目标和数据混在一起。

```text
stages/
├── stage0/
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── README.md
│   ├── src/
│   ├── configs/
│   └── experiments/
├── stage1/
│   ├── data/
│   ├── configs/
│   ├── experiments/
│   └── src/
└── stage2/
```

`stage0` 用于验证最小管线，`stage1` 用于建立标准 Transformer 基线，`stage2` 将用于构造受控推理评估集。每个阶段保留自己的数据处理产物和实验目录；真正稳定且跨阶段复用的工具，后续再抽到公共目录。
