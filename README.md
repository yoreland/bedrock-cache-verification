# Bedrock Claude Prompt Caching Verification

POC to verify AWS Bedrock Claude prompt caching behavior via the Converse API and Messages API (invoke_model).

## What This Tests

| Test | Scenario | API |
|------|----------|-----|
| **1** | System prompt caching (single checkpoint) | Converse |
| **2** | Tools + System caching (two checkpoints) | Converse |
| **3** | Multi-turn message history caching | Converse |
| **4** | No `cachePoint` baseline | Converse |
| **5** | Cache invalidation on system prompt change | Converse |
| **6** | Tools change — cascading cache invalidation | Converse |
| **7A** | Tools change with explicit `cache_control` | Messages |
| **7B** | Tools change without markers (Simplified Cache) | Messages |

## Quick Start

```bash
pip install boto3

# Run all tests (use inference profile ID, not bare model ID)
python bedrock_cache_poc.py --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 --region us-east-1

# Run a specific test
python bedrock_cache_poc.py --test 6 --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 --region us-east-1
```

> **Note:** Requires AWS credentials with Bedrock access. Use an inference profile ID (e.g. `us.anthropic.claude-sonnet-4-5-20250929-v1:0`) rather than a bare model ID.

## Test Results (Claude Sonnet 4.5)

### Test 6: Tools Change — Converse API with `cachePoint`

Setup: tools and system both have `cachePoint` markers. A third tool is added in calls 6c/6e to change the tools definition.

| Call | Action | Cache Read | Cache Write | Result |
|------|--------|-----------|-------------|--------|
| 6a | Original tools + system | 0 | 9,074 | 📝 WRITE |
| 6b | Same (unchanged) | 9,074 | 0 | ✅ HIT |
| 6c | **Changed tools**, same system + messages | 4,712 | 6,653 | ⚠️ Partial HIT |
| 6d | Back to original tools | 9,074 | 0 | ✅ HIT |
| 6e | Changed tools again | 11,365 | 0 | ✅ HIT |

**Observations:**
- 6c shows **partial cache hit** (read=4,712 ≈ system prompt size). When tools change, the system `cachePoint` still matches because its content is identical. Only the tools portion is re-cached.
- Each `cachePoint` is evaluated independently — changing content before one checkpoint does not invalidate checkpoints whose content is unchanged.
- Multiple cache versions coexist within TTL. Switching between tool versions (6d, 6e) hits the respective cached version.

### Test 7A: Tools Change — Messages API with explicit `cache_control`

Same tool-switching pattern as Test 6, using Messages API (`invoke_model`) with explicit `cache_control` markers.

| Call | Action | Cache Read | Cache Write | Result |
|------|--------|-----------|-------------|--------|
| 7Aa | Original tools | 9,074 | 0 | ✅ HIT |
| 7Ab | Same (unchanged) | 9,074 | 0 | ✅ HIT |
| 7Ac | **Changed tools** | 11,365 | 0 | ✅ HIT |
| 7Ad | Back to original | 9,074 | 0 | ✅ HIT |

**Observations:**
- 7Aa immediately hits cache — **Converse API and Messages API share the same cache pool**. Test 6's cached entries are reused.
- 7Ac also hits because Test 6e already warmed the changed-tools cache.

### Test 7B: Tools Change — Messages API without markers (Simplified Cache)

Same pattern, but with **no `cache_control` markers** — testing whether Bedrock's Simplified Cache activates automatically.

| Call | Action | Cache Read | Cache Write | Input Tokens | Result |
|------|--------|-----------|-------------|--------------|--------|
| 7Ba | Original tools | 0 | 0 | 9,410 | ⚪ NONE |
| 7Bb | Same (unchanged) | 0 | 0 | 9,410 | ⚪ NONE |
| 7Bc | Changed tools | 0 | 0 | 11,701 | ⚪ NONE |
| 7Bd | Back to original | 0 | 0 | 9,410 | ⚪ NONE |

**Observations:**
- Zero cache activity across all calls. **Simplified Cache does not activate on Bedrock's Messages API (invoke_model)**, even for Sonnet 4.5.
- Without explicit `cache_control` markers, no caching occurs — all input tokens are billed at full price.

## Key Findings

1. **Each `cachePoint` is evaluated independently** — Changing tools does not fully invalidate the system cache. If the system prompt content is identical, its `cachePoint` still hits. This means the cache is more granular than pure prefix-matching: each checkpoint tracks its own content hash.

2. **Multiple cache versions coexist** — Within TTL, different tool configurations are cached independently. Switching back to a previously cached configuration still hits.

3. **Converse API and Messages API share the same cache pool** — A cache entry written via Converse API can be read by Messages API and vice versa, as long as the content and checkpoint positions match.

4. **Simplified Cache does not work on Bedrock** — Anthropic's Simplified Cache (auto-caching without explicit markers) is not available on Bedrock's `invoke_model` endpoint. You must use explicit `cache_control` (Messages API) or `cachePoint` (Converse API) markers.

5. **No markers = no caching** — Without explicit markers, identical requests produce zero cache activity regardless of model version.

6. **`cachePoint` placement matters** — In the Converse API, `cachePoint` must be placed inside the `content` array as a content block:
   ```json
   {
     "role": "assistant",
     "content": [
       {"text": "..."},
       {"cachePoint": {"type": "default"}}
     ]
   }
   ```

## Converse API vs Messages API

| | Bedrock Converse API | Bedrock Messages API | Anthropic Direct API |
|---|---|---|---|
| Marker | `{"cachePoint": {"type": "default"}}` | `{"cache_control": {"type": "ephemeral"}}` | `{"cache_control": {"type": "ephemeral"}}` |
| Simplified Cache | ❌ | ❌ | ✅ (Sonnet 4.5+) |
| Processing order | `tools → system → messages` | `tools → system → messages` | `tools → system → messages` |
| Min tokens (Sonnet) | 1,024 | 1,024 | 1,024 |
| Max checkpoints | 4 | 4 | 4 |
| Default TTL | 5 min (Sonnet 4) / 1 hour (Sonnet 4.5+) | Same | 5 min |
| Shared cache pool | ✅ Shared with Messages API | ✅ Shared with Converse API | Separate |

## Supported Models

- Claude Sonnet 4 / 4.5 / 4.6
- Claude Opus 4.5 / 4.6
- Claude Haiku 4.5
- Claude 3.5 Sonnet v2, Claude 3.7 Sonnet
- Amazon Nova (Pro, Lite, Premier)

## References

- [AWS Bedrock Prompt Caching Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
- [Anthropic Prompt Caching Docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
