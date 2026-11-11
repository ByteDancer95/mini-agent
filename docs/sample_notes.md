# 示例笔记

这是一个用于验证 `read_docs` 工具的普通 Markdown 文件。

## 内容

- 工具应通过 JSON Schema 向 LLM 描述自己。
- 工具结果要放回 context，供下一轮决策使用。
- 思考过程默认不进入 LLM context，只落到 trace / 审计日志。
