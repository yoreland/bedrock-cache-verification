#!/usr/bin/env python3
"""Verify Bedrock Claude Opus 5.5 prompt-caching token accounting.

This script makes two real Bedrock Converse API calls to Claude Opus 5.5 with an
identical, cache-marked prefix. The first call is expected to WRITE the cache
(cacheWriteInputTokens > 0) and the second call, issued immediately with the same
prefix, is expected to READ the cache (cacheReadInputTokens > 0).

It prints the raw usage dict from each response and produces a cache-read cost
ESTIMATE using the Anthropic list price of $0.20 per 1,000,000 cache-read tokens.

Notes
-----
- Bedrock does not return dollar amounts in the API response. The dollar figure
  printed here is an estimate based on the published Anthropic price. The true
  billed cost must be confirmed the next day via Cost Explorer / CUR by filtering
  the usagetype that looks like "...cache-read-tokens-..." for this model, where:
      billed_cost / cache_read_tokens * 1e6  ~=  0.20
- The AWS Price List Bulk API and Pricing API do not expose Opus 5.5 or any
  structured claude cache-read line item, so programmatic price verification from
  a pricing data source is not possible. Hence: real call -> observe tokens ->
  reconcile against the bill.
"""

import json
import sys

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError

REGION = "us-west-2"

# Preferred inference profile, then fallback. Bedrock Anthropic large models
# generally require an inference profile id rather than the bare base model id
# for on-demand invocation.
CANDIDATE_MODEL_IDS = [
    "us.anthropic.claude-opus-5-5",       # US cross-region inference profile
    "global.anthropic.claude-opus-5-5",   # global inference profile (fallback)
]

# Anthropic published list price for cache-read input tokens.
CACHE_READ_PRICE_PER_TOKEN = 0.20 / 1_000_000  # USD per token ($0.20 / 1e6)

# A long, deterministic block of context placed before the cachePoint. It must
# exceed the model's minimum cacheable token threshold to trigger a cache write.
# Opus-class models require a fairly large minimum, so we build a big block.
_PARAGRAPH = (
    "In distributed systems, prompt caching lets a provider reuse the computed "
    "state of a repeated prefix across requests, so that the tokens forming that "
    "prefix are billed at a reduced cache-read rate on subsequent calls instead "
    "of the full input rate. This document exists purely to occupy a large, "
    "stable span of tokens so that the caching threshold is comfortably exceeded "
    "and a cache write is triggered on the first call, followed by a cache read "
    "hit on the second identical call. "
)


def build_long_context() -> str:
    """Return a deterministic block comfortably above the cache threshold."""
    blocks = []
    for i in range(120):
        blocks.append(f"Section {i:03d}. " + _PARAGRAPH)
    return "\n".join(blocks)


def build_system(long_context: str):
    """System field for Converse: long cached prefix, then a cachePoint."""
    return [
        {"text": long_context},
        {"cachePoint": {"type": "default"}},
    ]


def build_messages(user_question: str):
    return [
        {
            "role": "user",
            "content": [{"text": user_question}],
        }
    ]


def call_converse(client, model_id, system, messages):
    return client.converse(
        modelId=model_id,
        system=system,
        messages=messages,
        # NOTE: Opus 5.5 deprecates the `temperature` parameter, so it is
        # intentionally omitted from inferenceConfig.
        inferenceConfig={"maxTokens": 64},
    )


def pick_and_invoke(system, messages):
    """Try each candidate model id; return (model_id, first_response) on success.

    Raises the last ClientError if all candidates fail.
    """
    client = boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        config=Config(retries={"max_attempts": 3, "mode": "standard"}),
    )
    last_error = None
    for model_id in CANDIDATE_MODEL_IDS:
        try:
            print(f"[info] Trying inference profile: {model_id}")
            resp = call_converse(client, model_id, system, messages)
            print(f"[info] Success with inference profile: {model_id}")
            return client, model_id, resp
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "Unknown")
            msg = e.response.get("Error", {}).get("Message", str(e))
            print(f"[warn] {model_id} failed: {code}: {msg}")
            last_error = e
            # Retry the fallback for access/support related errors only.
            if code in (
                "AccessDeniedException",
                "ValidationException",
                "ResourceNotFoundException",
            ):
                continue
            raise
    assert last_error is not None
    raise last_error


def usage_dict(response):
    """Return the raw usage dict from a Converse response, as-is."""
    return response.get("usage", {})


def print_usage(label, usage):
    print(f"\n===== {label} usage (raw) =====")
    print(json.dumps(usage, indent=2, default=str))
    print(
        "  inputTokens            = {}\n"
        "  outputTokens           = {}\n"
        "  cacheReadInputTokens   = {}\n"
        "  cacheWriteInputTokens  = {}".format(
            usage.get("inputTokens"),
            usage.get("outputTokens"),
            usage.get("cacheReadInputTokens"),
            usage.get("cacheWriteInputTokens"),
        )
    )


def explain_access_error(e: ClientError):
    code = e.response.get("Error", {}).get("Code", "Unknown")
    msg = e.response.get("Error", {}).get("Message", str(e))
    print("\n" + "=" * 60)
    print(f"ERROR: {code}")
    print(msg)
    print("=" * 60)
    if code == "AccessDeniedException":
        print(
            "\nNext steps:\n"
            "  - Claude Opus 5.5 model access is likely not enabled for this\n"
            "    account/region. Open the Bedrock console -> Model access in\n"
            f"    region {REGION} and request access to Anthropic Claude Opus 5.5.\n"
            "  - Confirm the execution role has bedrock:InvokeModel on the\n"
            "    inference profile ARN(s).\n"
            "  - Both us.anthropic.claude-opus-5-5 and\n"
            "    global.anthropic.claude-opus-5-5 were attempted."
        )
    elif code == "ValidationException":
        print(
            "\nNext steps:\n"
            "  - The request shape or cachePoint placement may be invalid, or\n"
            "    on-demand invocation of the base model id may be unsupported.\n"
            "  - This script uses inference profile ids, which is the supported\n"
            "    path for Anthropic large models on Bedrock."
        )


def main() -> int:
    long_context = build_long_context()
    approx_chars = len(long_context)
    print(f"[info] Region: {REGION}")
    print(f"[info] Cached prefix length: {approx_chars} chars "
          f"(~{approx_chars // 4} tokens, rough 4 chars/token heuristic)")

    system = build_system(long_context)

    # ---- Call 1: expect a cache WRITE ----
    try:
        client, model_id, resp1 = pick_and_invoke(
            system, build_messages("Reply with the single word: one.")
        )
    except ClientError as e:
        explain_access_error(e)
        return 2
    except BotoCoreError as e:
        print(f"\n[error] BotoCoreError: {e}")
        return 3

    usage1 = usage_dict(resp1)
    print_usage("CALL 1 (expect cacheWriteInputTokens > 0)", usage1)

    # ---- Call 2: same prefix + cachePoint -> expect a cache READ ----
    try:
        resp2 = call_converse(
            client, model_id, system,
            build_messages("Reply with the single word: two."),
        )
    except ClientError as e:
        explain_access_error(e)
        return 2
    except BotoCoreError as e:
        print(f"\n[error] BotoCoreError: {e}")
        return 3

    usage2 = usage_dict(resp2)
    print_usage("CALL 2 (expect cacheReadInputTokens > 0)", usage2)

    # ---- Cost estimate + reconciliation guidance ----
    cache_read_tokens = usage2.get("cacheReadInputTokens") or 0
    # ESTIMATE ONLY: Anthropic list price $0.20 per 1e6 cache-read tokens.
    est_cache_read_cost = cache_read_tokens * CACHE_READ_PRICE_PER_TOKEN

    print("\n" + "=" * 60)
    print("CACHE READ COST ESTIMATE (NOT a billed amount)")
    print("=" * 60)
    print(f"  model id used            : {model_id}")
    print(f"  call 2 cacheReadInputTokens : {cache_read_tokens}")
    print(f"  Anthropic list price     : $0.20 per 1,000,000 cache-read tokens")
    print(f"  estimated cache-read cost: ${est_cache_read_cost:.8f}")
    print(
        "\n  NOTE: This dollar figure is an ESTIMATE derived from the published\n"
        "  Anthropic price. Bedrock does not return dollar cost in the API\n"
        "  response. To confirm the REAL cache-read unit price, wait ~1 day and\n"
        "  use Cost Explorer / CUR filtered by the usagetype containing\n"
        "  'cache-read-tokens' for this model. The reconciliation check is:\n"
        "      billed_cost / cache_read_tokens * 1e6  ~=  0.20"
    )

    cache_hit = cache_read_tokens > 0
    print("\n" + "=" * 60)
    print(f"CACHE HIT CONFIRMED: {cache_hit} "
          f"(call 2 cacheReadInputTokens = {cache_read_tokens})")
    print("=" * 60)

    # Emit a machine-readable results block for downstream capture.
    results = {
        "region": REGION,
        "model_id_used": model_id,
        "call1_usage": usage1,
        "call2_usage": usage2,
        "cache_read_tokens_call2": cache_read_tokens,
        "cache_read_price_per_million_usd": 0.20,
        "estimated_cache_read_cost_usd": est_cache_read_cost,
        "cache_hit": cache_hit,
    }
    print("\n----- RESULTS_JSON_BEGIN -----")
    print(json.dumps(results, indent=2, default=str))
    print("----- RESULTS_JSON_END -----")

    return 0 if cache_hit else 1


if __name__ == "__main__":
    sys.exit(main())
