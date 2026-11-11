# AI Prompt 与问题解决记录

> 这份文档记录**真实开发轨迹**，包括走过的弯路、被推翻的方案和我自己判断错的地方。
> 之前有一版是 30 条工整的「问题→解决」表格，看着漂亮但像事后包装；招聘方要这份文档
> 是为了区分「会让 AI 干活」和「会被 AI 带跑」，所以保留失败与返工比罗列成果更有价值。

---

## 一、开发阶段我给 AI 的 Prompt

**Prompt 1｜拆模块**
```
把需求拆成最小可运行的模块清单。约束：不依赖 langgraph/openhands/openclaw，
核心 Runtime 自己实现；LLM 必须是可替换组件；每个模块给出单一职责、
输入输出数据结构、以及可写的单元测试点。<贴题目原文>
```
产出十层拆分，并定下「Runtime 只认识 AgentOutput 与 ToolResult」这条解耦边界。

**Prompt 2｜设计输出协议**
```
设计一个不依赖厂商 function calling 的结构化输出协议，让模型自主判断
「直接回复」还是「调用工具」，能表达思考过程、并行工具调用、最终答案。
要求单个 JSON 对象、可被容错解析（模型常带 ```json 围栏或前后解释文字）。
```
产出 `prompt.py::PROTOCOL_SPEC` 与 `parser.py` 的多级容错。

**Prompt 3｜context 取舍**
```
ReAct loop 里，用户输入、工具调用、工具结果、思考过程分别该不该进入下一轮
LLM context？给出取舍理由、放置位置、基础压缩策略，并说明纯对话追问和
带工具追问如何共用同一份历史。
```
产出 README §3；核心结论是 **thought 只进审计与 trace，不进 LLM context**。

**Prompt 4｜零密钥可运行性**
```
仓库不能写任何 API Key，但 clone 下来要能演示完整 loop。设计一个 LLM 抽象，
使真实模型、脚本化 mock、本地确定性替身可互换；列出覆盖多 session 隔离 /
轮次上限 / 解析失败 / 工具异常的测试点。
```
产出 `LLMClient` Protocol + 三个实现与 `tests/`。

---

## 二、固化进 Runtime 的系统提示词（AI Prompt 本体）

由 `core/prompt.py::build_system_prompt()` **每轮动态生成**，工具目录来自 `registry.schemas()`，
所以新增工具不需要改提示词。

```
你是一个在工具循环（tool loop）中工作的助手 Mini-Agent。
你需要先判断是否已有足够信息回答，还是必须调用工具获取信息。

# 可用工具（JSON Schema 目录）
下面是全部已注册工具，字段名/类型/必填项以 parameters 为准：
<registry.schemas() 的 JSON 序列化>

# 输出协议（必须严格遵守）
你的每一次回复只能是 **一个** JSON 对象，不要输出 Markdown 代码块，不要输出任何额外文字。
两种合法形状，二选一：

1) 需要调用工具时：
{"thought": "简要说明你为什么需要调用这些工具（1-2 句）",
 "action": {"type": "tool",
            "tool_calls": [{"name": "工具名", "arguments": {"参数名": 值}}]}}

2) 可以直接回答时：
{"thought": "简要说明你已经掌握足够信息（1-2 句）",
 "action": {"type": "final", "content": "给用户的最终回答"}}

规则：
- 由你自己判断「直接回复」还是「调用工具」，判断依据只有上面的工具 Schema 与当前对话。
- arguments 必须符合对应工具 parameters JSON Schema 的类型与 required 约束。
- 一次可以同时调用多个互不依赖的工具，放进同一个 tool_calls 数组。
- 工具结果会以 role=tool 的消息返回给你；看到结果后，要么继续调用工具，要么给出最终答案。
- 严禁编造工具返回结果；严禁输出协议之外的字段或解释性文字。
- 最终答案使用与用户相同的语言。
```

辅助提示：`FORCE_FINAL_HINT`（达到轮次上限，禁止再调工具）、`REPAIR_HINT`
（输出不可解析时作为**临时** system 消息注入，不写进会话历史）。
`protocol="native"` 时改用简版提示词，Schema 改由 Chat Completions 的 `tools` 参数传递。

---

## 三、走过的弯路 / 被推翻的方案

| # | 当时怎么想 | 为什么错 | 最后改成 |
| --- | --- | --- | --- |
| 1 | 待办用模块级 `TODOS = []` 最简单 | 两个窗口立刻串台，正是题目要考的隔离点 | 取消全部全局可变状态，todo 只读写 `ctx.state`（会话私有） |
| 2 | calculator 直接 `eval(expression)` | `__import__('os').system(...)` 可任意执行 | `ast.parse(mode="eval")` + 节点白名单，拒绝属性访问/推导式 |
| 3 | 把 thought 原样回放进下一轮 context，显得"有推理" | token 上涨、模型复读上一步推理 | 默认剥离，只落 `_audit` + `Trace`；Policy 可打开 |
| 4 | 压缩只在「调用工具」分支后触发 | 纯对话长会话永远不压缩，history 无限增长 | **每轮 `run()` 开头也检查**（这次由不变量测试抓出来） |
| 5 | parser 严格 `json.loads` 即可 | 真实模型会带代码围栏、前后解释、半截 JSON | 剥围栏 → 括号配对提取 → 纯文本降级，五级容错 |
| 6 | `SessionManager.autosave` 方法名直观 | 与同名 bool 实例属性冲突，属性遮蔽方法，`bool` 不可调用 | 改名 `maybe_save` |
| 7 | session_id 直接当持久化文件名 | `../` 可穿越出目录 | 文件名用 `sha256(session_id)[:24]` |
| 8 | `ToolCall(call_id=None)` 表示"无 id" | 与 `str` 类型不符，序列化后下游要特判 | 有条件构造，无 id 时用 dataclass 默认值 |
| 9 | `read_docs` 判 `".." in path` 就够 | Windows 反斜杠与绝对路径能绕过 | `resolve()` 后必须满足 `root in target.parents` |
| 10 | 自研 JSON 协议和厂商 function calling 二选一 | 换端点就得改 Runtime | `LLMResponse` 同时带 `raw` 与 `native_tool_calls`，parser 优先 native |
| 11 | Python 3.14 沿用 `ast.Num` / `ast.Str` | 已废弃 | 统一 `ast.Constant` 并排除 bool |

---

## 四、AI 给错 / 我判断错的地方

* **AI 建议「加一个具名 Agent 让交付更具体」** → 我照做了，还配了独立 `main()` 和四轮脚本演示。
  直到确认这是**公司笔试题**才意识到：题目工具清单本身就是通用能力验收，这一层不命中评分点。
  最终保留类、删掉入口。
* **AI 建议「生成一份逐帧实录给评委看」** → 生成了 295 行 `RUN_TRANSCRIPT.md` 并提交。
  后来删掉：生成物不该进版本库，按需 `--save` 更可信，也少 295 行噪音。
* **我自己最大的判断错**：把「用户每一轮质疑」当成「需求变更」，于是每轮都加一层 ——
  具名整机 → 逐帧实录 → `--check-llm` → 评分点索引 → fake HTTP 端到端，四轮累计 4 个提交、
  1000 行文档。正确做法是先问清评分场景（比赛 demo 还是招聘笔试），再决定加不加。
  详见下一节。
* **措辞不够诚实**：早期把 `HeuristicLLM` 叫「零密钥演示模式」，容易让人误以为它是 AI。
  已收紧为「**离线协议验证替身，不声称是真实模型**」，并在 CLI 横幅、`describe()`、
  README 三处标注。
* **测试偏结构断言**：早期 129 个用例里不少在检查「某字段在不在 / trace 有几个事件」，
  而 parser 只被我自己构造的完美 JSON 测过。这轮补了 20 种真实模型坏输出语料与不变量测试。

---

## 五、关键问题解决（按机制分组，不再罗列 30 条）

| 组 | 问题 | 现在怎么处理 | 回归 |
| --- | --- | --- | --- |
| loop | 模型反复调工具不收敛 | `max_tool_rounds` → `force_final` → 确定性兜底 | `test_runtime.TestLoopLimit` |
| loop | 工具抛异常打断整轮 | `registry.execute` 全捕获，转 `ToolResult(ok=False)` 回喂 | `test_robustness.test_tool_exception_never_escapes_the_loop` |
| 解析 | 输出不是协议 JSON | 修复提示重试一次 → 纯文本降级 | `test_parser`、`test_robustness` |
| context | 压缩后追问失去指代 | 保证最近一条 `user` 不被折进 digest | `test_context` |
| 隔离 | 两个窗口共享状态 | 会话私有 `state`，文件名哈希 | `--demo`、`test_robustness` 交错不变量 |
| LLM | 无 key 跑不起来 / 密钥泄漏 | 三后端 + 只读环境变量 + HTTP 错误不打印 headers | `test_llm`、`test_openai_e2e`、密钥扫描 |

---

## 六、过度工程与减法（这一轮做的事）

**峰值体量**（第 3 轮结束时）：源码 3219 行 / 23 个 `.py`，测试 1717 行，
文档 1007 行（README 479 + 本文件 226 + 生成实录 295），4 个入口共 611 行。
题目叫「**最小可用** Agent」—— 面试官 15–30 分钟看不完这些。

| 砍掉 / 合并 | 原因 |
| --- | --- |
| `demo.py` 独立入口 → 并入 `cli.py --demo`；`agent.py` 的 `main()` / `_repl` / 四轮脚本删除 | 4 个入口各有 argparse/main，概念重复；`--demo` 已覆盖 |
| README 479 → ~190 行 | 只留运行 / 系统设计 / **memory 召回与放置** / 测试 / 取舍；评分点表压到附录 |
| 删除提交的 `docs/RUN_TRANSCRIPT.md` | 生成物不进版本库，`walkthrough --save` 按需生成 |
| 本文件从「30 条工整问题表」改成真实轨迹 | 招聘方要的是协作过程与判断力，不是包装过的成果清单 |

**没有砍的**：Runtime 主循环、5 个工具、Schema 注册、多 session 隔离、context 管理、trace、
异常分层 —— 这些是评分点本体。**新增的是替换而非叠加**：
20 种真实模型坏输出语料 + 不变量/fuzz 测试（随机畸形流必收敛、工具异常不外溢、
context 有界、交错会话不串台），以及顺手修掉的「纯对话不压缩」bug。

结论：对笔试题，写清「**砍了什么、代价是什么、为什么**」（README §6）比再多写 500 行代码
更能体现工程判断力。

---

## 七、复现验证

```bash
python -m mini_agent.cli --demo                  # 双窗口隔离：全部通过
python -m mini_agent.cli --check-llm            # 自检当前是不是真实 LLM
python -m mini_agent.walkthrough                # 逐帧查看一轮决策链路
python -m unittest discover -s tests -v         # 135 个用例
python -m unittest tests.test_robustness -v     # 坏输出语料 + 不变量
python -m unittest tests.test_openai_e2e -v     # 本地 fake 端点 + 真实 HTTP
```
