# Bedrock Claude Prompt Caching Verification

POC to verify AWS Bedrock Claude prompt caching (`cachePoint`) behavior via the Converse API.

## What This Tests

| Test | Scenario | Expected |
|------|----------|----------|
| **1** | System prompt caching | 1st call → WRITE, 2nd call → READ |
| **2** | Tools + System (two checkpoints) | 1st → WRITE, 2nd → READ all |
| **3** | Multi-turn message history caching | Incremental cache growth across turns |
| **4** | No `cachePoint` baseline | Zero cache activity (control group) |
| **5** | Cache invalidation on prefix change | Modified → new WRITE, revert → READ original |
| **6** | Tools change → cascading invalidation | Changed tools invalidates system & messages cache |

## Quick Start

```bash
pip install boto3

# Run all tests
python bedrock_cache_poc.py --region us-east-1

# Run a specific test
python bedrock_cache_poc.py --test 1 --region us-east-1

# Use a different model
python bedrock_cache_poc.py --model us.anthropic.claude-3-7-sonnet-20250219-v1:0
```

> **Note:** Requires AWS credentials with Bedrock access. Use an inference profile ID (e.g. `us.anthropic.claude-sonnet-4-20250514-v1:0`) rather than a bare model ID.

## Sample Output

```
[08:39:30] [INFO] TEST 1: System prompt caching (single checkpoint)
[08:39:34] [INFO] --- 1a: First call (expect WRITE) ---
[08:39:34] [INFO]   Input tokens:       11
[08:39:34] [INFO]   Cache WRITE tokens: 4358
[08:39:34] [INFO]   Cache READ tokens:  0
[08:39:34] [INFO]   📝 CACHE WRITE: Wrote 4358 tokens to cache

[08:39:39] [INFO] --- 1b: Second call (expect READ) ---
[08:39:39] [INFO]   Input tokens:       11
[08:39:39] [INFO]   Cache WRITE tokens: 0
[08:39:39] [INFO]   Cache READ tokens:  4358
[08:39:39] [SUCCESS]   ✅ CACHE HIT! Read 4358 tokens from cache
```

## Key Findings

1. **Cache metrics are in the response** — `usage.cacheReadInputTokens` / `cacheWriteInputTokens` are returned directly; no need for inference logging.

2. **`cachePoint` placement matters** — In the Converse API, `cachePoint` must be placed inside the `content` array as a content block, not at the message level.
   ```json
   {
     "role": "assistant",
     "content": [
       {"text": "..."},
       {"cachePoint": {"type": "default"}}
     ]
   }
   ```

3. **Caching is strict prefix-based** — If any content block changes in the middle, only the prefix up to the change is matched. The system writes a new cache entry for the full new prefix.

4. **Multiple cache versions coexist** — Within TTL, different prefix versions are cached independently. Switching back to a previously cached prefix still hits.

5. **No `cachePoint` = no caching** — The Converse API requires explicit `cachePoint` markers. Without them, identical requests produce zero cache activity.

6. **Tools change = cascading invalidation** — Because caching is strict prefix-based (`tools → system → messages`), modifying tools invalidates the cache for system and messages too, even if they're identical. This is why stable tool definitions are critical for cache efficiency.

## Converse API vs Anthropic API

| | Bedrock Converse API | Anthropic API |
|---|---|---|
| Marker | `{"cachePoint": {"type": "default"}}` | `{"cache_control": {"type": "ephemeral"}}` |
| Placement | Content block in `system` / `content` arrays | Same |
| Processing order | `tools → system → messages` | `tools → system → messages` |
| Min tokens (Sonnet) | 1,024 | 1,024 |
| Max checkpoints (Claude) | 4 | 4 |
| Default TTL | 5 min (Sonnet 4), 1 hour (Sonnet 4.5/Opus 4.5) | 5 min |

## Supported Models

- Claude Sonnet 4 / 4.5 / 4.6
- Claude Opus 4.5
- Claude Haiku 4.5
- Claude 3.5 Sonnet v2, Claude 3.7 Sonnet
- Amazon Nova (Pro, Lite, Premier)

## References

- [AWS Bedrock Prompt Caching Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
- [Anthropic Prompt Caching Docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
