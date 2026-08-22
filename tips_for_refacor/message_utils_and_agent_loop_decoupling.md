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
