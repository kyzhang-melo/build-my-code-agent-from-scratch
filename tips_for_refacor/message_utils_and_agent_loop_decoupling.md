# 给 myAgentHarness 重构的提醒：尽早解耦消息协议与 Agent Loop

## 为什么要在项目早期处理这件事

在 myCodeAgent-v0 中，会话历史最初直接保存 OpenAI Responses API 风格的
扁平 items。一次模型响应中的 assistant text、reasoning、function call 和
function call output 会在 agent loop 的不同代码分支里分别追加到 history。

这种设计在功能较少时很直观，但它把三个本应独立的概念绑定在了一起：

1. 模型或 provider 返回的 wire format；
2. harness 内部的会话存储格式；
3. agent loop 的控制流程和 append 时机。

最终产生过两类 thinking-mode 400 错误：reasoning 与 function call 的顺序被
破坏，以及 assistant text 插入 function call 和 tool output 之间。更麻烦的
是，错误历史可能在很多轮之后回放时才被 provider 拒绝，因此很难定位。

修复时不仅需要改 agent loop，还必须同步适配消息转换、session persistence、
resume sanitization、context compaction、subagent、eval 和大量测试。这说明模块
虽然在文件层面已经拆分，但“历史消息 schema”仍然是跨模块共享的隐式协议。

## myAgentHarness 应采用的边界

不要让 agent loop 直接把 provider items 当作内部 history。

建议从一开始就建立以下数据流：

```text
provider response
        ↓
message adapter / builder
        ↓
provider-neutral logical messages
        ↓
session persistence / resume / compaction
        ↓
provider serializer
        ↓
provider request items
```

一次模型响应应当作为一条完整 assistant message 存入 history，其内部保存有序
blocks，例如：

```python
{
    "role": "assistant",
    "content": [
        {"type": "reasoning", ...},
        {"type": "text", ...},
        {"type": "tool_call", ...},
    ],
}
```

工具结果不是 assistant response 的一部分，应作为独立 tool messages，连续紧跟
在完整 assistant message 后面：

```text
assistant[reasoning, text, call A, call B]
tool_result A
tool_result B
```

## 模块职责建议

### Agent loop

Agent loop 只负责 orchestration：发送请求、接收响应、调用 message builder、执行
工具、追加完整逻辑消息，以及判断是否继续循环。它不应该决定 reasoning、text
或 tool-call 在 provider 回放 payload 中的具体排列方式。

### Message utils / protocol adapter

消息模块应成为 provider 协议的唯一边界，负责：

- 将一次 provider response 按原序构造成一条 assistant message；
- 保存 call ID、provider item ID、reasoning signature 等配对信息；
- 从逻辑 history 序列化出 provider request items；
- 验证 call/result exchange 是否完整；
- 处理跨模型或跨 provider 的兼容与降级；
- 迁移旧消息 schema。

转换函数应尽可能是纯函数。序列化只做格式映射，不应在回放时重新猜测或调整
历史顺序。

### Persistence 和 compaction

session store 应持久化逻辑消息，而不是 provider wire items。compaction 必须在
完整 logical turn 或 tool exchange 边界切分，不能把 assistant tool call 与其
tool results 分到边界两侧。

## 必须显式维护的不变量

- 一次模型响应只能生成一个顶层 assistant history record；
- assistant blocks 保持 provider response 的原始相对顺序；
- tool results 必须属于前一条 assistant message 中的 calls；
- 同批 tool results 连续出现，中间不能插入 user 或 assistant message；
- call IDs 非空且唯一，result IDs 不得缺失、重复或未知；
- 异步工具的完成顺序不能改变持久化顺序；
- provider 配对 ID 与对应 block 一起保存，而不是依靠其他列表推断；
- 跨 runtime 时不得盲目回放其他 provider 的 reasoning、signature 或 item ID；
- context compaction 和 resume sanitization 不得拆断一个 tool exchange。

## 应避免的设计

- 在 `run_one_turn()` 的多个分支中分别 append text、reasoning、call 和 output；
- 使用 `output_text` 的存在与否临时决定历史 item 的位置；
- 让 persistence、compaction 和 eval 各自直接解析 provider-specific dict；
- 在 resume 或序列化阶段静默重排无法确定原始位置的 items；
- 仅使用单个 call 的固定 fixture 测试回放顺序，忽略并行 calls；
- 切换模型时只删除 reasoning，却继续回放旧 provider 的结构化 call/output。

## 建议的早期测试

除了示例测试，应尽早增加 message round-trip property test：

```text
provider response
→ logical assistant message
→ provider serialization
```

测试应验证 provider-originated blocks 的相对顺序和配对 ID 不变，并覆盖：

- reasoning、text 与 tool call 的多种排列；
- 单个和并行 tool calls；
- tool results 异步完成但稳定入列；
- malformed arguments；
- incomplete、duplicate 和 unknown results；
- session round-trip 和旧版本迁移；
- compaction 边界；
- 相同 runtime 与跨 runtime replay；
- parent agent 与 subagent 使用同一消息协议。

## 最重要的结论

文件拆分不等于真正解耦。如果多个模块都直接读取 `msg["type"]`、假设
`msg["content"]` 一定是字符串，或者理解 provider item 的配对规则，那么消息
schema 仍然是一个隐式、全项目耦合的接口。

myAgentHarness 应在早期把逻辑消息模型和协议转换 API 明确定义出来，让 agent
loop 依赖稳定的 message abstraction，而不是依赖某个 provider 当前的 wire
format。这样未来增加 thinking model、切换 provider、引入持久化或 context
compaction 时，改动会集中在协议适配层，而不会再次牵动整个 harness。

## 社区佐证：这是 Responses API agent 开发的普遍坑（2026-08 调研）

这个坑并非本项目独有。GitHub 与 OpenAI 官方论坛有大量同类事故，说明
"配对约束真实存在但未文档化"是这个生态的普遍现状：

### 官方 SDK 仓库的未解 issue

- openai/openai-python#3009 与 openai/openai-node#1791：标题即
  "undocumented reasoning+message pairing constraint breaks multi-turn
  conversations"。核心内容：reasoning 与 message items 必须在 input 中
  作为连续对出现，但文档从未说明；最常见的模式——过滤
  `response.output` 只保留 messages——会静默产生孤儿 items，下一轮 400。
  该问题曾导致 OpenClaw（64.9k forks）在 gpt-5.3-codex 上崩溃，最终他们
  不得不添加 `downgradeOpenAIReasoningBlocks()` 来剥离孤儿 reasoning。
- openai/openai-python#3008：把 `response.output` 经 `model_dump()` 原样
  回传，第二轮 400——SDK 会编造 `status: None` 等字段，API 拒绝未知 null
  参数。同样是"忠实回放"与"序列化边界"没有分离的问题。

### OpenAI 官方论坛的同款报错

- "'function_call' was provided without its required 'reasoning' item"
  （本项目 bug 1 的 OpenAI 原版报错）：从 o3-mini 换 o4-mini 后必现；
  社区 workaround 是在 function_call 前补一个空 reasoning item。
- "400 No tool call found for function call output with call_id"
  （本项目 bug 2 的 OpenAI 原版）：多个帖子，甚至 store: true 且正确
  echo 也会偶发。
- "Item 'rs_...' of type 'reasoning' was provided without its required
  following item"：OpenAI 官方模型上 reasoning↔message 也有配对约束，
  不只 DeepSeek 这类第三方 thinking 模型。
- openclaw/openclaw#12885：store: false 时回放含 rs_ id 的历史直接 404
  "Item not found"——reasoning item ID 的生命周期还受服务端存储策略影响。

### 官方文档只写"应该做"，没写"违反会怎样"

- Reasoning models 指南：pass back any reasoning items returned with
  the last function call（语气是 recommend，实际是硬约束）。
- Cookbook（Handling Function Calls with Reasoning Models）：it is
  essential that we preserve any reasoning and function call responses
  in our conversation history。
- Conversation state 指南：Replaying the complete output keeps
  reasoning items and assistant phase values intact。

### 三条被反复验证的规律

1. 配对约束真实存在但未文档化：reasoning → function_call →
   function_call_output 的顺序与配对在 OpenAI 官方模型、OpenRouter 的
   DeepSeek 等多处生效，报错文案各异但根因相同。
2. 最常见的触发模式就是对 `response.output` 做过滤或重排后再回放——
   正是扁平追加式 harness 的天然形态。
3. 成熟项目的解法殊途同归：OpenClaw 加 downgradeOpenAIReasoningBlocks()，
   pi 用 thinkingSignature 完整保存原始 item，本项目用逻辑消息 + 忠实
   序列化 + 跨 runtime textualize——本质都是"存储 provider 无关结构，
   回放时完整还原或显式降级，绝不猜测顺序"。
