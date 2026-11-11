# 一轮 Agent 决策的完整实录

- 问题：`查一下广州今天的天气，顺便记个待办：明天给大棚番茄喷药`
- 决策后端：`HeuristicLLM` —— 离线协议验证替身（只用于验证 loop / 解析 / 工具 / 隔离的数据形态，**不是真实模型输出**）
- 组件：与 `AgentRuntime.run()` 完全同构（同一套 prompt / context / parser / registry）

---

## 帧 1｜交给模型的工具 Schema 目录

模型只能看到这张表，据此自己判断「直接回复」还是「调用工具」。

### tool_schemas

```text
[
  {
    "name": "calculator",
    "description": "安全计算一个数学表达式并返回数值。支持 + - * / // % ** 、括号、sqrt/abs/round/min/max/pow/log/sin/cos/floor/ceil 等函数与 pi/e 常量。需要算数、单位换算、成本估算时使用；不支持变量赋值或任意代码。",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "要计算的表达式，例如 '12*(3+4)' 或 'sqrt(144)/2'",
          "minLength": 1,
          "maxLength": 200
        }
      },
      "required": [
        "expression"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "search",
    "description": "在本地知识语料中检索与 query 相关的资料，返回标题与摘要（mock 实现，结果可复现）。需要背景知识、资料引用或不确定事实时使用。",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "检索关键词或自然语言问题",
          "minLength": 1
        },
        "top_k": {
          "type": "integer",
          "description": "返回条数",
          "minimum": 1,
          "maximum": 10
        }
      },
      "required": [
        "query"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "todo",
    "description": "管理当前会话的待办清单。action 支持 add（新增）、list（列出）、complete（标记完成）、remove（删除）、clear（清空）。待办按会话隔离，不同窗口互不影响。",
    "parameters": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": [
            "add",
            "list",
            "complete",
            "remove",
            "clear"
          ],
          "description": "要执行的操作"
        },
        "item": {
          "type": "string",
          "description": "add 时的待办内容",
          "minLength": 1
        },
        "priority": {
          "type": "string",
          "enum": [
            "low",
            "normal",
            "high"
          ],
          "description": "add 时的优先级，默认 normal"
        },
        "id": {
          "type": "integer",
          "description": "complete/remove 时指定待办编号",
          "minimum": 1
        }
      },
      "required": [
        "action"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "weather",
    "description": "查询指定城市的当日天气（mock 数据，用于演示工具扩展）。涉及天气/温度/是否适合施药时使用。",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "城市名，例如 '广州'",
          "minLength": 1
        },
        "date": {
          "type": "string",
          "description": "日期，留空表示今天"
        }
      },
      "required": [
        "city"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "read_docs",
    "description": "读取本地 docs/ 目录中的文档内容（如周报模板、写作规范）。需要模板或内部规范时使用。",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string",
          "description": "相对 docs/ 的文件名，如 'weekly_report_template.md'",
          "minLength": 1
        },
        "max_chars": {
          "type": "integer",
          "description": "最多读取字符数",
          "minimum": 100,
          "maximum": 20000
        }
      },
      "required": [
        "path"
      ],
      "additionalProperties": false
    }
  }
]
```

## 帧 2｜发给模型的 messages（system prompt 摘要）

### system prompt（前 900 字）

```text
你是一个在工具循环（tool loop）中工作的助手 Mini-Agent。
你需要先判断是否已有足够信息回答，还是必须调用工具获取信息。

# 可用工具（JSON Schema 目录）
下面是全部已注册工具，字段名/类型/必填项以 parameters 为准：
[
  {
    "name": "calculator",
    "description": "安全计算一个数学表达式并返回数值。支持 + - * / // % ** 、括号、sqrt/abs/round/min/max/pow/log/sin/cos/floor/ceil 等函数与 pi/e 常量。需要算数、单位换算、成本估算时使用；不支持变量赋值或任意代码。",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "要计算的表达式，例如 '12*(3+4)' 或 'sqrt(144)/2'",
          "minLength": 1,
          "maxLength": 200
        }
      },
      "required": [
        "expression"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "search",
    "description": "在本地知识语料中检索与 query 相关的资料，返回标题与摘要（mock 实现，结果可复现）。需要背景知识、资料引用或不确定事实时使用。",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "strin
…（共 3983 字）
```

### messages 角色序列

```text
0. [system] 你是一个在工具循环（tool loop）中工作的助手 Mini-Agent。 你需要先判断是否已有足够信息回答，还是必须调用工具获取信息。  # 可用工具（JS
1. [user] 查一下广州今天的天气，顺便记个待办：明天给大棚番茄喷药
```

## 帧 3｜模型第 1 次原始输出（未做任何加工）

### LLM 原文

```text
{"thought": "当前信息不足以直接回答，需要调用下列工具获取事实。", "action": {"type": "tool", "tool_calls": [{"name": "weather", "arguments": {"city": "广州"}}, {"name": "todo", "arguments": {"action": "add", "item": "明天给大棚番茄喷药"}}]}}
```

## 帧 4｜解析结果：思考过程 / 工具调用 / 决策

### parse

```text
source      = json
thought     = 当前信息不足以直接回答，需要调用下列工具获取事实。
decision    = tool
tool_calls  = [{"name": "weather", "arguments": {"city": "广州"}, "call_id": "d3186dc72d"}, {"name": "todo", "arguments": {"action": "add", "item": "明天给大棚番茄喷药"}, "call_id": "2e3240c785"}]
warning     = None
```

## 帧 5｜Runtime 执行工具（Schema 校验 + 异常隔离）

### tool results

```text
weather({"city": "广州"})
  -> ok=True elapsed=0.038ms
  -> {"ok": true, "result": {"city": "广州", "date": "today", "condition": "多云转阵雨", "temperature": 26, "wind": "东南风 2 级", "humidity": 78, "advice": "湿度偏高，建议延后叶面施药并加强通风", "mock": true}}

todo({"action": "add", "item": "明天给大棚番茄喷药"})
  -> ok=True elapsed=0.051ms
  -> {"ok": true, "result": {"action": "add", "message": "已新增待办 #1: 明天给大棚番茄喷药", "session_id": "walkthrough", "todos": [{"id": 1, "item": "明天给大棚番茄喷药", "priority": "normal", "done": false}]}}
```

## 帧 6｜工具结果回灌后的 context

思考过程 **不** 进入 LLM context；工具结果以 `role=tool` 紧跟决策之后。

### messages 角色序列

```text
0. [system] 你是一个在工具循环（tool loop）中工作的助手 Mini-Agent。 你需要先判断是否已有足够信息回答，还是必须调用工具获取信息。  # 可用工具（JSON Schema 目录） 下面是全部已注册工具，字段名/类型/必填项以 parameters 为准： [   {   
1. [user] 查一下广州今天的天气，顺便记个待办：明天给大棚番茄喷药
2. [assistant] {"action": {"type": "tool", "tool_calls": [{"name": "weather", "arguments": {"city": "广州"}}, {"name": "todo", "arguments": {"action": "add",
3. [tool] <tool_result name="weather" call_id="d3186dc72d">{"ok": true, "result": {"city": "广州", "date": "today", "condition": "多云转阵雨", "temperature":
4. [tool] <tool_result name="todo" call_id="2e3240c785">{"ok": true, "result": {"action": "add", "message": "已新增待办 #1: 明天给大棚番茄喷药", "session_id": "walk
```

## 帧 7｜模型看到工具结果后的第 2 次输出

### LLM 原文

```text
{"thought": "已收到工具结果，信息足够，可以给出最终答案。", "action": {"type": "final", "content": "已按你的要求完成：\n- 广州天气：多云转阵雨，26°C，湿度偏高，建议延后叶面施药并加强通风\n- 当前待办（1 项）：1. 明天给大棚番茄喷药"}}
```

## 帧 8｜最终答案

> 已按你的要求完成：
- 广州天气：多云转阵雨，26°C，湿度偏高，建议延后叶面施药并加强通风
- 当前待办（1 项）：1. 明天给大棚番茄喷药

---

## 附录｜同一个问题交给真正的 `AgentRuntime.run()` 后的 trace

### trace

```text
trace run=f2fe9b5d2e18 session=walkthrough-session
  [t00] run_start      
  [t00] user_input     
  [t01] llm_call       
  [t01] llm_response   
  [t01] thought        
  [t01] parse          source=json warning=None
  [t01] decision       tool
  [t01] tool_call      weather({"city": "广州"})
  [t01] tool_result    weather ok=True 0.023ms {"ok": true, "result": {"city": "广州", "date": "today", "condition": "多云转阵雨", "temperature": 26, "win
  [t01] tool_call      todo({"action": "add", "item": "明天给大棚番茄喷药"})
  [t01] tool_result    todo ok=True 0.044ms {"ok": true, "result": {"action": "add", "message": "已新增待办 #1: 明天给大棚番茄喷药", "session_id": "walkthroug
  [t02] llm_call       
  [t02] llm_response   
  [t02] thought        
  [t02] parse          source=json warning=None
  [t02] decision       final
  [t02] final          已按你的要求完成：
- 广州天气：多云转阵雨，26°C，湿度偏高，建议延后叶面施药并加强通风
- 当前待办（1 项）：1. 明天给大棚番茄喷药
  [t00] session_save
```

status = `ok`  turns = `2`  工具调用数 = `2`

> 复现：`python -m mini_agent.walkthrough --save <path>`

> 本文件由离线替身生成，仅用于展示协议形态；配置 `.env` 后加 `--llm openai` 即可生成真实模型版本。