# 阶段 0 数据目录

数据处理遵循以下顺序：

```text
data/raw/       原始复制文件，只读保存
      ↓
data/processed/ 清洗、格式化和 tokenizer 处理后的文件
      ↓
training        训练脚本读取的最终数据
```

`raw/` 中的文件来自 `K:\MyLm\nanoGPT\data\catgirl_prepare_json\data.json` 以及对应的 `train.txt`、`val.txt`。后续处理时不要直接修改原始文件。
