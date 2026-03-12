#!/usr/bin/env python3
"""
Cross-process cache verification.
Tests whether Bedrock prompt cache is a global content-addressable KV store.

Process A: Write cache with fixed seed content
Process B: Same content → HIT = global KV, MISS = process-isolated
Process C: Different seed, same structure → should MISS

Usage:
  python test_cross_process.py --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 --region us-east-1
"""

import boto3
import json
import time
import argparse
import subprocess
import sys
import os
from datetime import datetime


def log(msg, level="INFO"):
    pid = os.getpid()
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [PID={pid}] [{level}] {msg}")


def call_messages_api(client, model_id, system, messages, label):
    """Call Messages API and return cache metrics."""
    log(f"  → {label}")
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "system": system,
        "messages": messages,
    }
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    result = json.loads(response["body"].read())
    usage = result.get("usage", {})
    metrics = {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
        "cache_write": usage.get("cache_creation_input_tokens", 0),
        "cache_read": usage.get("cache_read_input_tokens", 0),
    }
    status = "✅ HIT" if metrics["cache_read"] > 0 else \
             "📝 WRITE" if metrics["cache_write"] > 0 else "⚪ NONE"
    log(f"  ← {status} read={metrics['cache_read']} write={metrics['cache_write']} input={metrics['input']}")
    return metrics


def build_content(seed):
    """Build deterministic content from a seed string."""
    # Padding to exceed 1024 token minimum
    from bedrock_cache_poc import generate_padding
    padding = generate_padding(1200)
    
    system = [{
        "type": "text",
        "text": f"You are a helpful assistant. Seed={seed}\n{padding}",
        "cache_control": {"type": "ephemeral"}
    }]
    
    messages = []
    for i in range(5):
        messages.append({"role": "user", "content": [
            {"type": "text", "text": f"[seed={seed}][Turn {i+1}] Tell me about item {i+1}."}
        ]})
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": f"[seed={seed}][Turn {i+1}] Item {i+1} is great."}
        ]})
    # cache_control on last assistant
    messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    # final user question
    messages.append({"role": "user", "content": [
        {"type": "text", "text": f"[seed={seed}] Summarize."}
    ]})
    return system, messages


def run_as_child(model_id, region, seed, label):
    """Run in a subprocess to ensure completely independent process."""
    script = f"""
import sys
sys.path.insert(0, '{os.path.dirname(os.path.abspath(__file__))}')
from test_cross_process import call_messages_api, build_content, log
import boto3, json

client = boto3.client("bedrock-runtime", region_name="{region}")
system, messages = build_content("{seed}")
import copy
m = call_messages_api(client, "{model_id}", copy.deepcopy(system), copy.deepcopy(messages), "{label}")
# Output result as JSON for parent to parse
print("RESULT:" + json.dumps(m))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    
    # Parse result
    for line in result.stdout.split("\n"):
        if line.startswith("RESULT:"):
            return json.loads(line[7:])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    SHARED_SEED = "cross-process-test-fixed-2026"
    DIFFERENT_SEED = "cross-process-test-OTHER-2026"

    log("=" * 60)
    log("CROSS-PROCESS CACHE VERIFICATION")
    log("=" * 60)
    log(f"Shared seed: {SHARED_SEED}")
    log(f"Different seed: {DIFFERENT_SEED}")

    # Step 1: Process A — write cache (in THIS process)
    log("\n── STEP 1: Process A (this process) — establish cache ──")
    client = boto3.client("bedrock-runtime", region_name=args.region)
    system, messages = build_content(SHARED_SEED)
    import copy
    m_a = call_messages_api(client, args.model, copy.deepcopy(system), 
                            copy.deepcopy(messages), "Process A: write cache")
    time.sleep(3)

    # Step 2: Process A again — verify warm (same process)
    log("\n── STEP 2: Process A again — verify warm (same process) ──")
    m_a2 = call_messages_api(client, args.model, copy.deepcopy(system),
                             copy.deepcopy(messages), "Process A: verify warm")
    time.sleep(2)

    # Step 3: Process B — DIFFERENT process, SAME content
    log("\n── STEP 3: Process B (child process) — same seed ──")
    log("  Spawning subprocess with same seed...")
    m_b = run_as_child(args.model, args.region, SHARED_SEED, "Process B: same seed")
    time.sleep(2)

    # Step 4: Process C — DIFFERENT process, DIFFERENT content
    log("\n── STEP 4: Process C (child process) — different seed ──")
    log("  Spawning subprocess with different seed...")
    m_c = run_as_child(args.model, args.region, DIFFERENT_SEED, "Process C: different seed")

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    results = [
        ("Process A (write)", m_a),
        ("Process A (verify)", m_a2),
        ("Process B (same seed, diff process)", m_b),
        ("Process C (diff seed, diff process)", m_c),
    ]
    for label, m in results:
        if m:
            status = "✅ HIT" if m["cache_read"] > 0 else \
                     "📝 WRITE" if m["cache_write"] > 0 else "⚪ NONE"
            log(f"  {label}: {status} read={m['cache_read']} write={m['cache_write']}")
        else:
            log(f"  {label}: ❌ FAILED")

    log("\n  CONCLUSION:")
    if m_b and m_b["cache_read"] > 0:
        log("  Process B (same content, different process): ✅ CACHE HIT")
        log("  → Bedrock cache is GLOBAL content-addressable (not process/session-isolated)")
    else:
        log("  Process B: CACHE MISS → cache may be process/session-scoped")

    if m_c and m_c["cache_read"] == 0:
        log("  Process C (different content): ✅ CACHE MISS (as expected)")
        log("  → Different content = different cache entry (content-hash based)")
    elif m_c and m_c["cache_read"] > 0:
        log("  Process C: ⚠️ UNEXPECTED HIT — cache may not be content-based?")


if __name__ == "__main__":
    main()
