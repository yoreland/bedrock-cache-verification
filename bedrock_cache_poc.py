#!/usr/bin/env python3
"""
Bedrock Claude Prompt Caching POC
=================================
Demonstrates cache behavior with system/tools/messages checkpoints.
Uses Converse API to verify cache read/write via response metrics.

Prerequisites:
  pip install boto3
  AWS credentials configured (us-east-1 or us-west-2 with Claude access)

Usage:
  python3 bedrock_cache_poc.py [--model MODEL_ID] [--region REGION]
"""

import boto3
import json
import time
import argparse
import sys
from datetime import datetime


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def create_client(region):
    return boto3.client("bedrock-runtime", region_name=region)


def extract_cache_metrics(response):
    """Extract cache-related metrics from Converse API response."""
    usage = response.get("usage", {})
    return {
        "inputTokens": usage.get("inputTokens", 0),
        "outputTokens": usage.get("outputTokens", 0),
        "cacheReadInputTokens": usage.get("cacheReadInputTokens", 0),
        "cacheWriteInputTokens": usage.get("cacheWriteInputTokens", 0),
    }


def print_metrics(label, metrics):
    """Pretty-print cache metrics."""
    log(f"--- {label} ---")
    log(f"  Input tokens:       {metrics['inputTokens']}")
    log(f"  Output tokens:      {metrics['outputTokens']}")
    log(f"  Cache WRITE tokens: {metrics['cacheWriteInputTokens']}")
    log(f"  Cache READ tokens:  {metrics['cacheReadInputTokens']}")
    
    if metrics['cacheReadInputTokens'] > 0:
        log(f"  ✅ CACHE HIT! Read {metrics['cacheReadInputTokens']} tokens from cache", "SUCCESS")
    elif metrics['cacheWriteInputTokens'] > 0:
        log(f"  📝 CACHE WRITE: Wrote {metrics['cacheWriteInputTokens']} tokens to cache", "INFO")
    else:
        log(f"  ⚠️  NO CACHE activity (tokens below minimum or caching not triggered)", "WARN")


# ─── Generate padding text to meet minimum token requirements ───
def generate_padding(target_tokens=1200):
    """Generate enough text to exceed minimum cache checkpoint token requirement.
    Most Claude models need 1024 tokens minimum per checkpoint.
    """
    # ~4 chars per token on average
    lines = []
    for i in range(target_tokens // 10):
        lines.append(f"Rule {i+1}: When handling queries about AWS service #{i+1}, "
                     f"always consider security best practices, cost optimization, "
                     f"and operational excellence according to the Well-Architected Framework.")
    return "\n".join(lines)


SYSTEM_PROMPT_PADDING = generate_padding(1200)

SYSTEM_PROMPT = f"""You are an expert AWS Solutions Architect assistant. 
You help customers with cloud architecture, cost optimization, and migration strategies.
Always provide concise, actionable advice.

{SYSTEM_PROMPT_PADDING}
"""

TOOL_DEFINITIONS = [
    {
        "toolSpec": {
            "name": "get_ec2_pricing",
            "description": "Retrieve EC2 instance pricing for a specified instance type and region. "
                          "Returns on-demand, reserved, and spot pricing. " + generate_padding(600),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "instance_type": {"type": "string", "description": "EC2 instance type (e.g., m5.xlarge)"},
                        "region": {"type": "string", "description": "AWS region (e.g., us-east-1)"}
                    },
                    "required": ["instance_type", "region"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "get_service_limits",
            "description": "Check AWS service quotas and limits for a specific service and region. " + generate_padding(600),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string", "description": "AWS service name"},
                        "region": {"type": "string", "description": "AWS region"}
                    },
                    "required": ["service"]
                }
            }
        }
    }
]


def call_converse(client, model_id, system, tools, messages, test_name):
    """Call Bedrock Converse API and return cache metrics."""
    log(f"\n{'='*60}")
    log(f"TEST: {test_name}")
    log(f"{'='*60}")
    
    kwargs = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": {"maxTokens": 200},
    }
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["toolConfig"] = {"tools": tools}
    
    try:
        response = client.converse(**kwargs)
        metrics = extract_cache_metrics(response)
        print_metrics(test_name, metrics)
        
        # Print model response (truncated)
        output = response.get("output", {}).get("message", {}).get("content", [])
        if output:
            text = output[0].get("text", "")[:150]
            log(f"  Response: {text}...")
        
        return metrics
    except Exception as e:
        log(f"  ❌ ERROR: {e}", "ERROR")
        return None


# ═══════════════════════════════════════════════════
# TEST SCENARIOS
# ═══════════════════════════════════════════════════

def test_1_system_only_cache(client, model_id):
    """Test 1: Cache only system prompt, verify cache hit on 2nd call."""
    log("\n\n" + "█" * 60)
    log("TEST 1: System prompt caching (single checkpoint)")
    log("█" * 60)
    log("Expected: 1st call → cache WRITE, 2nd call → cache READ")
    
    system = [
        {"text": SYSTEM_PROMPT},
        {"cachePoint": {"type": "default"}}  # Checkpoint on system
    ]
    
    # Call 1: Should write to cache
    messages_1 = [{"role": "user", "content": [{"text": "What is Amazon S3?"}]}]
    m1 = call_converse(client, model_id, system, None, messages_1, "1a: First call (expect WRITE)")
    
    time.sleep(2)
    
    # Call 2: Same system, different question → should READ from cache
    messages_2 = [{"role": "user", "content": [{"text": "What is Amazon EC2?"}]}]
    m2 = call_converse(client, model_id, system, None, messages_2, "1b: Second call (expect READ)")
    
    return m1, m2


def test_2_system_and_tools_cache(client, model_id):
    """Test 2: Cache both tools and system with separate checkpoints."""
    log("\n\n" + "█" * 60)
    log("TEST 2: Tools + System caching (two checkpoints)")
    log("█" * 60)
    log("Expected: 1st → WRITE, 2nd (same tools+system) → READ")
    
    # Tools with checkpoint
    tools = TOOL_DEFINITIONS + [{"cachePoint": {"type": "default"}}]
    
    # System with checkpoint
    system = [
        {"text": SYSTEM_PROMPT},
        {"cachePoint": {"type": "default"}}
    ]
    
    # Call 1
    messages_1 = [{"role": "user", "content": [{"text": "How much does m5.xlarge cost?"}]}]
    m1 = call_converse(client, model_id, system, tools, messages_1, "2a: First call (expect WRITE)")
    
    time.sleep(2)
    
    # Call 2: Same tools + system, different question
    messages_2 = [{"role": "user", "content": [{"text": "What are the limits for Lambda?"}]}]
    m2 = call_converse(client, model_id, system, tools, messages_2, "2b: Second call (expect READ)")
    
    return m1, m2


def test_3_message_history_cache(client, model_id):
    """Test 3: Cache conversation history in messages."""
    log("\n\n" + "█" * 60)
    log("TEST 3: Message history caching (multi-turn)")
    log("█" * 60)
    log("Expected: Growing conversation, system+history cached")
    
    system = [
        {"text": SYSTEM_PROMPT},
        {"cachePoint": {"type": "default"}}
    ]
    
    # Turn 1: Initial question
    messages_turn1 = [
        {"role": "user", "content": [{"text": "I need to migrate 200K cores from Alibaba Cloud to AWS. What's the best approach?"}]}
    ]
    m1 = call_converse(client, model_id, system, None, messages_turn1, "3a: Turn 1 (initial)")
    
    time.sleep(2)
    
    # Turn 2: Add history + cache checkpoint in last assistant content + new question
    messages_turn2 = [
        {"role": "user", "content": [{"text": "I need to migrate 200K cores from Alibaba Cloud to AWS. What's the best approach?"}]},
        {"role": "assistant", "content": [
            {"text": "For migrating 200K cores from Alibaba Cloud to AWS, I recommend a phased approach: 1) Assessment phase - categorize workloads by complexity and business criticality. 2) Pilot phase - migrate 5-10% of non-critical workloads first. 3) Migration waves - group remaining workloads into waves based on dependencies. Key considerations include network connectivity (Direct Connect vs VPN), data transfer strategy, and application compatibility testing."},
            {"cachePoint": {"type": "default"}}  # Cache checkpoint at end of assistant content
        ]},
        {"role": "user", "content": [
            {"text": "What about the cost comparison between Alibaba Cloud Singapore and AWS Singapore?"}
        ]}
    ]
    m2 = call_converse(client, model_id, system, None, messages_turn2, "3b: Turn 2 (history + new question)")
    
    time.sleep(2)
    
    # Turn 3: Same history prefix, different new question
    messages_turn3 = [
        {"role": "user", "content": [{"text": "I need to migrate 200K cores from Alibaba Cloud to AWS. What's the best approach?"}]},
        {"role": "assistant", "content": [
            {"text": "For migrating 200K cores from Alibaba Cloud to AWS, I recommend a phased approach: 1) Assessment phase - categorize workloads by complexity and business criticality. 2) Pilot phase - migrate 5-10% of non-critical workloads first. 3) Migration waves - group remaining workloads into waves based on dependencies. Key considerations include network connectivity (Direct Connect vs VPN), data transfer strategy, and application compatibility testing."},
            {"cachePoint": {"type": "default"}}  # Same cache checkpoint
        ]},
        {"role": "user", "content": [
            {"text": "What about the reliability comparison? I heard Alibaba Cloud Singapore had an LSE incident."}
        ]}
    ]
    m3 = call_converse(client, model_id, system, None, messages_turn3, "3c: Turn 3 (same history, new question → expect READ)")
    
    return m1, m2, m3


def test_4_no_cache_control_baseline(client, model_id):
    """Test 4: Baseline WITHOUT cache_control — verify no caching occurs."""
    log("\n\n" + "█" * 60)
    log("TEST 4: No cache_control baseline (expect NO cache activity)")
    log("█" * 60)
    
    system = [{"text": SYSTEM_PROMPT}]  # No cachePoint!
    
    messages_1 = [{"role": "user", "content": [{"text": "What is Amazon S3?"}]}]
    m1 = call_converse(client, model_id, system, None, messages_1, "4a: No cache_control (expect NO cache)")
    
    time.sleep(2)
    
    messages_2 = [{"role": "user", "content": [{"text": "What is Amazon EC2?"}]}]
    m2 = call_converse(client, model_id, system, None, messages_2, "4b: No cache_control again (expect NO cache)")
    
    return m1, m2


def test_5_cache_invalidation(client, model_id):
    """Test 5: Verify cache invalidation when prefix changes."""
    log("\n\n" + "█" * 60)
    log("TEST 5: Cache invalidation (modify system → expect MISS)")
    log("█" * 60)
    
    # Call 1: Original system
    system_v1 = [
        {"text": SYSTEM_PROMPT},
        {"cachePoint": {"type": "default"}}
    ]
    messages = [{"role": "user", "content": [{"text": "What is S3?"}]}]
    m1 = call_converse(client, model_id, system_v1, None, messages, "5a: Original system (WRITE)")
    
    time.sleep(2)
    
    # Call 2: Modified system → cache should MISS, new WRITE
    system_v2 = [
        {"text": SYSTEM_PROMPT + "\nAdditional rule: Always mention pricing."},
        {"cachePoint": {"type": "default"}}
    ]
    m2 = call_converse(client, model_id, system_v2, None, messages, "5b: Modified system (expect NEW WRITE, not READ)")
    
    time.sleep(2)
    
    # Call 3: Back to original system → should READ from first cache (if still in TTL)
    m3 = call_converse(client, model_id, system_v1, None, messages, "5c: Back to original (expect READ from first cache)")
    
    return m1, m2, m3


def test_6_tools_change_cascading_invalidation(client, model_id):
    """Test 6: Verify that modifying tools invalidates system AND messages cache.
    
    Prompt caching is strict prefix-based: tools → system → messages.
    If tools change, everything downstream (system + messages) must re-cache,
    even if system and messages are identical.
    """
    log("\n\n" + "█" * 60)
    log("TEST 6: Tools change → cascading cache invalidation")
    log("█" * 60)
    log("Expected: Changing tools invalidates system & messages cache")
    log("This verifies the strict prefix-matching behavior:")
    log("  tools (changed) → system (same) → messages (same) = ALL MISS")
    
    # Shared system prompt with checkpoint
    system = [
        {"text": SYSTEM_PROMPT},
        {"cachePoint": {"type": "default"}}
    ]
    
    # Original tools with checkpoint
    tools_v1 = TOOL_DEFINITIONS + [{"cachePoint": {"type": "default"}}]
    
    # Modified tools: add a new tool (changes the tools prefix)
    tools_v2 = TOOL_DEFINITIONS + [
        {
            "toolSpec": {
                "name": "get_s3_storage_cost",
                "description": "Calculate S3 storage costs for a given amount of data and storage class. " + generate_padding(600),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "storage_gb": {"type": "number", "description": "Storage amount in GB"},
                            "storage_class": {"type": "string", "description": "S3 storage class (STANDARD, IA, GLACIER, etc.)"},
                            "region": {"type": "string", "description": "AWS region"}
                        },
                        "required": ["storage_gb"]
                    }
                }
            }
        },
        {"cachePoint": {"type": "default"}}
    ]
    
    # Same messages for all calls
    messages = [{"role": "user", "content": [{"text": "What is the cheapest EC2 option for a web server?"}]}]
    
    # Call 1: Original tools + system → expect WRITE for both
    m1 = call_converse(client, model_id, system, tools_v1, messages,
                       "6a: Original tools+system (expect WRITE)")
    
    time.sleep(2)
    
    # Call 2: Same everything → expect READ (verify cache is warm)
    m2 = call_converse(client, model_id, system, tools_v1, messages,
                       "6b: Same tools+system (expect READ — cache warm)")
    
    time.sleep(2)
    
    # Call 3: CHANGED tools, same system, same messages → expect WRITE (all cache miss)
    m3 = call_converse(client, model_id, system, tools_v2, messages,
                       "6c: Changed tools, same system+messages (expect WRITE — cascade invalidation)")
    
    time.sleep(2)
    
    # Call 4: Back to original tools → expect READ (original cache still in TTL)
    m4 = call_converse(client, model_id, system, tools_v1, messages,
                       "6d: Back to original tools (expect READ — original cache still valid)")
    
    time.sleep(2)
    
    # Call 5: Changed tools again → expect READ (v2 cache now warm too)
    m5 = call_converse(client, model_id, system, tools_v2, messages,
                       "6e: Changed tools again (expect READ — v2 cache now warm)")
    
    return m1, m2, m3, m4, m5


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Bedrock Claude Prompt Caching POC")
    parser.add_argument("--model", default="anthropic.claude-sonnet-4-20250514-v1:0",
                       help="Model ID (default: Claude Sonnet 4)")
    parser.add_argument("--region", default="us-east-1",
                       help="AWS region (default: us-east-1)")
    parser.add_argument("--test", type=int, choices=[1, 2, 3, 4, 5, 6],
                       help="Run specific test only (1-6)")
    args = parser.parse_args()
    
    log(f"Bedrock Prompt Caching POC")
    log(f"Model: {args.model}")
    log(f"Region: {args.region}")
    log(f"System prompt size: ~{len(SYSTEM_PROMPT.split())} words")
    
    client = create_client(args.region)
    
    all_results = {}
    
    tests = {
        1: ("System-only cache", test_1_system_only_cache),
        2: ("Tools + System cache", test_2_system_and_tools_cache),
        3: ("Message history cache", test_3_message_history_cache),
        4: ("No cache baseline", test_4_no_cache_control_baseline),
        5: ("Cache invalidation", test_5_cache_invalidation),
        6: ("Tools change cascading invalidation", test_6_tools_change_cascading_invalidation),
    }
    
    run_tests = [args.test] if args.test else [1, 2, 3, 4, 5, 6]
    
    for test_num in run_tests:
        name, func = tests[test_num]
        try:
            results = func(client, args.model)
            all_results[test_num] = results
        except Exception as e:
            log(f"Test {test_num} failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
    
    # Summary
    log("\n\n" + "═" * 60)
    log("SUMMARY")
    log("═" * 60)
    
    for test_num in run_tests:
        name = tests[test_num][0]
        if test_num in all_results:
            results = all_results[test_num]
            if results:
                log(f"\nTest {test_num}: {name}")
                for i, m in enumerate(results):
                    if m:
                        status = "✅ HIT" if m['cacheReadInputTokens'] > 0 else \
                                 "📝 WRITE" if m['cacheWriteInputTokens'] > 0 else \
                                 "⚪ NONE"
                        log(f"  Call {i+1}: {status} (read={m['cacheReadInputTokens']}, write={m['cacheWriteInputTokens']})")


if __name__ == "__main__":
    main()
