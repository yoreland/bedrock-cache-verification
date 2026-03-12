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
| **8** | Single checkpoint beyond 20 content blocks | Messages |
| **9A** | Simplified mode: modification detection | Messages |
| **9B** | Manual mode: checkpoint independence | Messages |
| **10** | Sliding window cache pattern | Messages |
| **Cross-process** | Global cache verification | Messages |

## Quick Start

```bash
pip install boto3

# Run all tests (use inference profile ID, not bare model ID)
python bedrock_cache_poc.py --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 --region us-east-1

# Run a specific test
python bedrock_cache_poc.py --test 9 --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 --region us-east-1

# Cross-process cache verification (separate script)
python test_cross_process.py --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 --region us-east-1
```

> **Note:** Requires AWS credentials with Bedrock access. Use an inference profile ID (e.g. `us.anthropic.claude-sonnet-4-5-20250929-v1:0`) rather than a bare model ID.

## Test Results (Claude Sonnet 4.5)

### Test 6: Tools Change — Converse API with `cachePoint`

Setup: tools and system both have `cachePoint` markers (2 checkpoints). A third tool is added in calls 6c/6e.

| Call | Action | Cache Read | Cache Write | Result |
|------|--------|-----------|-------------|--------|
| 6a | Original tools + system | 0 | 9,074 | 📝 WRITE |
| 6b | Same (unchanged) | 9,074 | 0 | ✅ HIT |
| 6c | **Changed tools**, same system + messages | 4,712 | 6,653 | ⚠️ Partial HIT |
| 6d | Back to original tools | 9,074 | 0 | ✅ HIT |
| 6e | Changed tools again | 11,365 | 0 | ✅ HIT |

**Observations:**
- 6c shows **partial cache hit** (read=4,712 ≈ system prompt size). With 2 checkpoints, changing tools does not invalidate the system checkpoint.
- Each `cachePoint` is evaluated independently.
- Multiple cache versions coexist within TTL.

### Test 7B: Messages API without markers

| Call | Action | Cache Read | Cache Write | Result |
|------|--------|-----------|-------------|--------|
| 7Ba–7Bd | Various | 0 | 0 | ⚪ NONE |

**No `cache_control` markers = zero cache activity.** Bedrock requires explicit markers.

### Test 8: Single Checkpoint Beyond 20 Content Blocks

One `cache_control` on last assistant block. Multi-block assistant messages (3 text blocks each).

| Call | Turns | Content Blocks | Cache Read | Cache Write | Result |
|------|-------|---------------|-----------|-------------|--------|
| 4t-first | 4 | 17 | 0 | 4,738 | 📝 WRITE |
| 4t-same | 4 | 17 | 4,738 | 0 | ✅ HIT |
| 5t-first | 5 | 21 | 4,738 | 107 | ✅ incremental |
| 7t-first | 7 | 29 | 4,845 | 177 | ✅ incremental |
| 10t-first | 10 | 41 | 5,022 | 266 | ✅ incremental |
| 10t-same | 10 | 41 | 5,288 | 0 | ✅ HIT |

Caching works correctly even at **41 content blocks**. Cache grows incrementally as conversation extends.

### Test 9: Simplified vs Manual Mode — Clean Verification

UUID-stamped content to guarantee no cache residue from prior runs.

**9A: Simplified Mode (1 checkpoint on last assistant only)**

| Call | Action | Cache Read | Cache Write | Result |
|------|--------|-----------|-------------|--------|
| 9Aa | Establish cache (31 blocks) | 0 | 5,346 | 📝 WRITE |
| 9Ab | Same | 5,346 | 0 | ✅ HIT |
| 9Ac | **Modify block #1** (early) | 0 | 5,330 | ❌ MISS |
| 9Ad | **Modify block #25** (late) | 0 | 5,330 | ❌ MISS |
| 9Ae | **Modified system** | 0 | 5,350 | ❌ MISS |
| 9Af | **Added tools** | 0 | 10,379 | ❌ MISS |
| 9Ag | Extended to 33 blocks | 5,346 | 65 | ✅ incremental |
| 9Ah | Back to original | 5,346 | 0 | ✅ HIT |

With 1 checkpoint: **any modification anywhere = full cache miss.** Only appending new content at the end produces incremental cache growth.

**9B: Manual Mode (2 checkpoints: system + last assistant)**

| Call | Action | Cache Read | Cache Write | Result |
|------|--------|-----------|-------------|--------|
| 9Ba | Establish cache (31 blocks) | 0 | 5,284 | 📝 WRITE |
| 9Bb | Same | 5,284 | 0 | ✅ HIT |
| 9Bc | **Modify block #1** (early) | **4,369** | 899 | ✅ **Partial HIT** |
| 9Bd | **Modify block #25** (late) | **4,369** | 899 | ✅ **Partial HIT** |
| 9Be | **Modified system** | 0 | 5,288 | ❌ MISS |
| 9Bf | Back to original | 5,284 | 0 | ✅ HIT |

With 2 checkpoints: **system checkpoint survives message changes** (read=4,369 ≈ system prompt size). Only changing the system prompt itself causes a full miss.

### Test 10: Sliding Window Cache Pattern

Maintain a conversation window of ~20 blocks. As new turns arrive, drop the oldest. 2 checkpoints (system + last assistant).

| Call | Window | Blocks | Cache Read | Cache Write | Result |
|------|--------|--------|-----------|-------------|--------|
| 10a | turns 1-10 | 21 | 0 | 4,958 | 📝 WRITE |
| 10b | same | 21 | 4,958 | 0 | ✅ HIT |
| 10c | **turns 3-10** (dropped 1-2) | 17 | **4,368** | 472 | ✅ system HIT |
| 10d | same | 17 | 4,840 | 0 | ✅ HIT |
| 10e | **turns 3-12** (slide +2) | 21 | **4,840** | 118 | ✅ incremental |
| 10f | same | 21 | 4,958 | 0 | ✅ HIT |
| 10g | **turns 5-14** (slide +2) | 21 | **4,368** | 590 | ✅ system HIT |
| 10h | same | 21 | 4,958 | 0 | ✅ HIT |

**Sliding window is viable.** When dropping early turns:
- System prompt cache (~4,368 tokens) **always hits** — the biggest cost saving
- Only the changed messages portion needs re-caching
- Appending new turns at the end is purely incremental

### Cross-Process Cache Verification

Separate script (`test_cross_process.py`) using independent subprocesses.

| Process | Content | PID | Cache Read | Cache Write | Result |
|---------|---------|-----|-----------|-------------|--------|
| A | seed=fixed | 3022935 | 0 | 4,620 | 📝 WRITE |
| A (verify) | same | 3022935 | 4,620 | 0 | ✅ HIT |
| **B** (child) | **same seed** | **3023055** | **4,620** | **0** | **✅ HIT** |
| **C** (child) | **different seed** | **3023098** | **0** | **4,620** | **📝 MISS** |

**Bedrock prompt cache is a global content-addressable KV store:**
- Same content + different process → HIT (not session/process-isolated)
- Different content + different process → MISS (content-hash indexed)
- Shared across all requests within the same AWS account + region + model

## Key Findings

1. **Simplified mode (1 checkpoint) vs Manual mode (2+ checkpoints) behave differently.** Simplified mode treats the entire prefix as one unit — any change = full miss. Manual mode evaluates each checkpoint independently — system cache survives message changes.

2. **Manual mode is strictly better for long conversations.** With 2 checkpoints (system + last assistant), the system prompt cache is protected even when conversation content changes or slides. This is critical for agentic and multi-turn use cases.

3. **Sliding window pattern works.** Drop old turns, keep recent ones, use 2 checkpoints. System cache (~4K+ tokens) always hits. Only the messages portion needs re-caching on each window slide.

4. **Cache is global and content-addressable.** Verified across independent processes. All requests in the same account/region/model share the cache. Multiple instances of a service with the same system prompt automatically benefit from shared caching.

5. **Converse API and Messages API share the same cache pool.** A cache entry written via one API can be read by the other.

6. **No markers = no caching on Bedrock.** Unlike Anthropic's direct API (which has Simplified Cache for Sonnet 4.5+), Bedrock requires explicit `cache_control` / `cachePoint` markers. Without them, zero cache activity occurs.

7. **Appending content is always incremental.** Adding new turns at the end of a conversation always produces prefix hit + incremental write, regardless of total block count (tested up to 41 blocks).

8. **Multiple cache versions coexist within TTL.** Switching between different tool configurations or conversation variants hits the respective cached version, as long as it's within TTL.

## Best Practices

Based on test results, the recommended caching strategy:

```
[tools + cachePoint] → [system + cachePoint] → [messages... last_assistant + cachePoint] → [new user]
```

- **Always use 2-3 explicit checkpoints** (not just 1) — protects stable content from changes in volatile content
- **Place checkpoints at stability boundaries**: tools (rarely change) → system (rarely changes) → last assistant (changes every turn)
- **For sliding windows**: keep system + tools cached, only slide the messages portion
- **Keep tools and system prompts stable** — changes to tools/system invalidate everything downstream

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
