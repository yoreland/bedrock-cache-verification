# Verify Bedrock Claude Opus 5.5 prompt-caching token accounting

This is a small standalone script that makes **real** Bedrock `Converse` API calls
to **Claude Opus 5.5** to verify prompt-caching token accounting, and in
particular to sanity-check the **cache-read** unit price against the published
Anthropic list price of **$0.20 per 1,000,000 tokens**.

## Why this approach

The AWS Price List Bulk API and the Pricing API freeze the Claude family at very
old model versions. They do not expose Opus 5.5 and contain no structured
`claude` cache-read line item, so the cache-read price cannot be verified from a
pricing data source programmatically. The only reliable path is:

1. Make a real call and observe the token usage (`cacheReadInputTokens`).
2. Reconcile the real dollar cost the next day via Cost Explorer / CUR.

## What the script does

1. Builds a long, deterministic context block (well above the model's minimum
   cacheable-token threshold) and marks it with a Converse `cachePoint`.
2. **Call 1** with that prefix -> expects a cache **write**
   (`cacheWriteInputTokens > 0`).
3. **Call 2** with the identical prefix + `cachePoint`, issued immediately ->
   expects a cache **read** hit (`cacheReadInputTokens > 0`).
4. Prints the raw `usage` dict from each response.
5. Computes a cache-read **cost estimate** = `cacheReadInputTokens * ($0.20 / 1e6)`.
   This is an estimate only; the API does not return dollar cost.
6. Handles `AccessDeniedException` / `ValidationException` with readable next steps.

It prefers the US cross-region inference profile `us.anthropic.claude-opus-5-5`
and falls back to `global.anthropic.claude-opus-5-5` on access/validation errors.
The inference profile actually used is printed. (Bedrock Anthropic large models
generally require an inference profile id rather than the bare base model id
`anthropic.claude-opus-5-5` for on-demand invocation.)

## Prerequisites

- AWS credentials with `bedrock:InvokeModel` permission on the Opus 5.5
  inference profile(s), in **account 077090643075**.
- Region **us-west-2** (the script forces this).
- Claude Opus 5.5 model access enabled in Bedrock for that account/region.
- Python 3.11 + `boto3` (this project uses an isolated `uv` venv).

## How to run

```bash
cd opus55-cache-verify

# one-time: create an isolated env with boto3
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python boto3

# run (region is forced to us-west-2 inside the script; env vars shown for clarity)
AWS_REGION=us-west-2 AWS_DEFAULT_REGION=us-west-2 .venv/bin/python verify_opus55_cache.py
```

The script is idempotent and safe to re-run. Each run makes two billable
inference calls (a few thousand cache/input tokens total; well under $0.10).

Exit codes: `0` = cache hit confirmed, `1` = ran but no cache read observed,
`2` = access/validation error, `3` = other boto error.

## Actual run result (real API call)

Executed against account `077090643075`, region `us-west-2`, using inference
profile **`us.anthropic.claude-opus-5-5`**. Full machine-readable output is in
[`results.json`](./results.json); raw console output is in `run_output.txt`.

**Call 1 usage (cache write):**

```json
{
  "inputTokens": 16,
  "outputTokens": 3,
  "totalTokens": 18260,
  "cacheReadInputTokens": 0,
  "cacheWriteInputTokens": 18241,
  "cacheDetails": [{ "ttl": "5m", "inputTokens": 18241 }]
}
```

**Call 2 usage (cache read hit):**

```json
{
  "inputTokens": 16,
  "outputTokens": 3,
  "totalTokens": 18260,
  "cacheReadInputTokens": 18241,
  "cacheWriteInputTokens": 0
}
```

**Cache hit: CONFIRMED.** Call 2 returned `cacheReadInputTokens = 18241`
(and `cacheWriteInputTokens = 0`), exactly matching the tokens written by call 1.

**Cache-read cost estimate (Anthropic list price, NOT a billed amount):**

```
cache_read_tokens (call 2) = 18241
estimate = 18241 * ($0.20 / 1e6) = $0.00364820
```

## How to confirm the REAL cache-read price ($0.20 / 1e6) from the bill

Bedrock does not return a dollar amount in the API response, so the figure above
is an estimate. To confirm the real unit price, wait ~1 day for billing to land,
then in **Cost Explorer** (or a CUR query):

1. Filter to service **Amazon Bedrock**, region **us-west-2**, account
   **077090643075**.
2. Group by **Usage Type** and find the usagetype for this model containing
   `cache-read-tokens` (e.g. something like `...-cache-read-tokens-...`).
3. Read the **usage quantity** (number of cache-read tokens, ~18,241 from this
   run plus any other traffic) and the **cost** for that usagetype.
4. Reconcile:

   ```
   billed_cost / cache_read_tokens * 1e6  ~=  0.20
   ```

   If it comes out near **$0.20**, the Bedrock cache-read unit price matches the
   Anthropic published price.

## Notes / gotchas

- **`temperature` is deprecated for Opus 5.5.** The Converse request omits it;
  including it returns `ValidationException: temperature is deprecated for this
  model`.
- Cache TTL observed was **5m** (`cacheDetails[].ttl`), so call 2 must follow
  call 1 promptly (the script does this in a single run).
- The cached prefix here is ~18,241 tokens, comfortably above the Opus caching
  minimum, which is why the first call triggered a write rather than being
  ignored.
