# Bedrock Claude Prompt Caching 验证

验证 AWS Bedrock Claude 的 prompt caching（`cachePoint`）机制，通过 Converse API 实测缓存行为。

## 缓存机制说明

### 工作原理

Bedrock prompt caching 基于**严格前缀匹配**。模型按 `tools → system → messages` 顺序处理输入，缓存从头开始匹配——只要前缀完全相同就能命中，前缀中任何位置发生变化，从变化点往后全部失效。

### 核心概念

**cachePoint 标记**

在 Converse API 中，通过 `{"cachePoint": {"type": "default"}}` 标记缓存检查点，放在 `system`、`toolConfig.tools`、或 message `content` 数组中作为 content block：

```json
{
  "system": [
    {"text": "你是一个 AWS 架构师助手...（大段 system prompt）"},
    {"cachePoint": {"type": "default"}}
  ]
}
```

```json
{
  "role": "assistant",
  "content": [
    {"text": "上一轮的回答..."},
    {"cachePoint": {"type": "default"}}
  ]
}
```

> ⚠️ Bedrock Converse API 用 `cachePoint`，Anthropic 原生 API 用 `cache_control`，参数名不同。

**处理顺序：tools → system → messages**

缓存前缀按此顺序拼接。如果你在 tools 和 system 各放一个 cachePoint，tools 的缓存在前、system 在后。修改 tools 会导致 system 缓存也失效（因为前缀变了）。

**最多 4 个检查点**

Claude 模型单次请求最多 4 个 cachePoint。超出的会被忽略。

**最小 token 要求**

| 模型 | 每个 checkpoint 最小 token |
|------|--------------------------|
| Claude Sonnet 4 / 3.7 / 3.5 v2 | 1,024 |
| Claude Opus 4.5 / Haiku 4.5 | 4,096 |

不满足最小 token 的 cachePoint 不会触发缓存写入。

**TTL（缓存存活时间）**

| 模型 | 默认 TTL |
|------|---------|
| Claude Sonnet 4 / 3.5 / 3.7 | 5 分钟 |
| Claude Sonnet 4.5 / Opus 4.5 / Haiku 4.5 | 1 小时 |

每次缓存命中（READ）会刷新 TTL。

**计费优势**

- Cache READ：输入 token 费用 **降低 90%**
- Cache WRITE：输入 token 费用 **增加 25%**
- 只要同一前缀被读取 2 次以上就划算

### 简化缓存管理（Simplified Cache Management）

Bedrock 还提供一种不需要手动放 cachePoint 的模式：系统自动从最后一个 user message 往前看约 20 个 content block boundary，尝试匹配已有缓存。这在 InvokeModel API 中**默认开启**，Converse API 需要显式 cachePoint。

### 多版本缓存共存

不同前缀的缓存在 TTL 内独立存在。比如 system prompt 修改后会写入新缓存，但改回原来的 system prompt 仍然能命中旧缓存（只要还在 TTL 内）。

---

## 测试方法

### 环境要求

```bash
pip install boto3
# AWS credentials 需要有 Bedrock 访问权限
```

### 运行方式

```bash
# 运行全部 5 个测试
python bedrock_cache_poc.py --region us-east-1

# 运行单个测试
python bedrock_cache_poc.py --test 1 --region us-east-1

# 指定模型（需要用 inference profile ID）
python bedrock_cache_poc.py --model us.anthropic.claude-3-7-sonnet-20250219-v1:0
```

### 5 个测试场景

**Test 1：System Prompt 缓存**
- Call 1：发送带 cachePoint 的 system prompt + 问题 → 预期 WRITE
- Call 2：相同 system prompt + 不同问题 → 预期 READ（命中缓存）

**Test 2：Tools + System 双检查点**
- Call 1：tools（带 cachePoint）+ system（带 cachePoint）+ 问题 → 预期 WRITE
- Call 2：相同 tools 和 system + 不同问题 → 预期 READ

**Test 3：多轮对话历史缓存**
- Turn 1：system（带 cachePoint）+ 问题 → 写入 system 缓存
- Turn 2：同 system + 历史消息（assistant content 末尾带 cachePoint）+ 新问题 → READ system + WRITE history
- Turn 3：同 system + 同历史 + 不同新问题 → READ 全部（system + history）

**Test 4：无 cachePoint 对照组**
- Call 1 & 2：完全不加 cachePoint，其他条件相同 → 预期零缓存活动

**Test 5：缓存失效验证**
- Call 1：原始 system prompt → WRITE
- Call 2：修改 system prompt → 新 WRITE（旧缓存失效）
- Call 3：改回原始 system prompt → READ（命中第一次的缓存）

### 验证指标

每次 API 调用的 response `usage` 字段直接返回缓存指标，**不需要开启 inference log**：

```json
{
  "usage": {
    "inputTokens": 11,
    "outputTokens": 200,
    "cacheReadInputTokens": 4358,
    "cacheWriteInputTokens": 0
  }
}
```

- `cacheWriteInputTokens > 0` → 写入了新缓存
- `cacheReadInputTokens > 0` → 命中了已有缓存
- 两者都为 0 → 没有缓存活动

### 实测结果

使用 `us.anthropic.claude-sonnet-4-20250514-v1:0`，us-east-1：

| Test | 场景 | Call 1 | Call 2 | Call 3 |
|------|------|--------|--------|--------|
| **1** | System 缓存 | 📝 WRITE 4358 | ✅ READ 4358 | — |
| **2** | Tools + System | 📝 WRITE 9074 | ✅ READ 9074 | — |
| **3** | 多轮历史 | ✅ READ 4358 | ✅ READ 4358 + 📝 WRITE 128 | ✅ READ 4486 |
| **4** | 无 cachePoint | ⚪ 无缓存 | ⚪ 无缓存 | — |
| **5** | 缓存失效 | ✅ READ 4358 | 📝 WRITE 4365 | ✅ READ 4358 |

### 关键发现

1. **缓存指标直接在 response 中**，不需要 CloudWatch 或 inference log
2. **cachePoint 必须放在 content 数组内部**，不能放在 message 级别
3. **多版本缓存共存**——Test 5 证明修改后改回原来仍然命中旧缓存
4. **Test 3 递增缓存**——Turn 2 读 system(4358) + 写 history(128)，Turn 3 读 system+history(4486)
5. **不加 cachePoint 就完全没有缓存**——Test 4 对照组确认

## 参考文档

- [AWS Bedrock Prompt Caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
