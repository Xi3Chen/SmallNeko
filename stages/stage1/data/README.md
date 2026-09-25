# 阶段 1 数据目录

阶段 1 的正式基线使用 UTF-8 字节级 tokenizer 和连续文本流。

把数据准备成以下文件：

```text
data/raw/train.txt
data/raw/val.txt
```

`train.txt` 和 `val.txt` 必须是 UTF-8 编码。对话数据建议使用下面的边界格式：

```text
<|im_start|>user
用户问题
<|im_start|>assistant
助手回答
<|im_end|>
```

也可以先用 `src/format_chat.py` 将 JSON、JSONL、`instruction/output`、
`question/answer`、OpenAI `messages` 或 ShareGPT `conversations` 格式转换成文本。

阶段 1 的 tokenizer 词表固定，不从数据统计字符，因此能够编码任意合法 Unicode 文本。
`processed/` 中的 `.bin` 文件和元数据由脚本生成，不要手工编辑。

当前目录中的数据如果只是来自阶段 0 的猫娘对话，只适合验证管线；正式基线应使用覆盖面更稳定、去重并经过质量筛选的语料。

本机默认抽取配置位于 `configs/data-mixture.json`，使用中文 Wikipedia 70%、OpenWebMath 28%、阶段 0 对话 2%。配置会从多个 parquet 分片抽取约 500 MB 文本到 `data/raw/`，原始分片不会提交到 Git。阶段 0 数据总量有限，因此扩容后降低其比例以避免重复采样。

准备正式数据时，建议至少保留以下信息：来源或文档 ID、原始语言、数据类型、清洗版本和训练/验证归属。验证集应按文档或来源划分，避免同一问题、同一文档的近重复内容同时出现在两个 split 中。

对于本项目的学习目标，可以分三层准备数据：先用几百万 token 的小数据检查代码，再用数千万 token 比较不同模型结构，最后在更大的固定语料上确认结论。字节 tokenizer 中一个中文字符通常占 3 个 token，所以应以脚本输出的 token 数记录训练预算，不能只按文件 MB 数比较。
