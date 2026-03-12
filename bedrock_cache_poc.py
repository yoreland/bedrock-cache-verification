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


# ─── Messages API helpers ───

# Map Converse API model IDs to Messages API model IDs
MESSAGES_API_MODEL_MAP = {
    "anthropic.claude-sonnet-4-20250514-v1:0": "anthropic.claude-sonnet-4-20250514-v1:0",
    "us.anthropic.claude-sonnet-4-20250514-v1:0": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
}


def converse_tool_to_messages_tool(tool_spec):
    """Convert Converse API toolSpec to Messages API tool format."""
    spec = tool_spec.get("toolSpec", {})
    return {
        "name": spec.get("name", ""),
        "description": spec.get("description", ""),
        "input_schema": spec.get("inputSchema", {}).get("json", {}),
    }


def call_messages_api(client, model_id, system, tools, messages, test_name,
                       cache_controls=None):
    """Call Bedrock Messages API (invoke_model) and return cache metrics.
    
    Args:
        cache_controls: dict of where to place cache_control markers.
            e.g. {"tools_last": True, "system_last": True, "last_assistant": True}
            If None, no explicit markers (tests Simplified Cache).
    """
    log(f"\n{'='*60}")
    log(f"TEST (Messages API): {test_name}")
    log(f"{'='*60}")

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "messages": messages,
    }

    if system:
        if isinstance(system, str):
            system_blocks = [{"type": "text", "text": system}]
        elif isinstance(system, list) and all(isinstance(s, str) for s in system):
            system_blocks = [{"type": "text", "text": s} for s in system]
        else:
            system_blocks = system  # already in block format
        
        if cache_controls and cache_controls.get("system_last") and system_blocks:
            system_blocks[-1]["cache_control"] = {"type": "ephemeral"}
        
        body["system"] = system_blocks

    if tools:
        msg_tools = [converse_tool_to_messages_tool(t) for t in tools if "toolSpec" in t]
        
        if cache_controls and cache_controls.get("tools_last") and msg_tools:
            msg_tools[-1]["cache_control"] = {"type": "ephemeral"}
        
        body["tools"] = msg_tools

    # Add cache_control to last assistant message if requested
    if cache_controls and cache_controls.get("last_assistant") and messages:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if content:
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

    try:
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        usage = result.get("usage", {})

        metrics = {
            "inputTokens": usage.get("input_tokens", 0),
            "outputTokens": usage.get("output_tokens", 0),
            "cacheWriteInputTokens": usage.get("cache_creation_input_tokens", 0),
            "cacheReadInputTokens": usage.get("cache_read_input_tokens", 0),
        }
        print_metrics(test_name, metrics)

        output = result.get("content", [])
        if output:
            text = output[0].get("text", "")[:150]
            log(f"  Response: {text}...")

        return metrics
    except Exception as e:
        log(f"  ❌ ERROR: {e}", "ERROR")
        return None


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


# ─── Extra tool for Messages API (same content, Messages API format) ───

EXTRA_TOOL_MESSAGES = {
    "name": "get_s3_storage_cost",
    "description": "Calculate S3 storage costs for a given amount of data and storage class. " + generate_padding(600),
    "input_schema": {
        "type": "object",
        "properties": {
            "storage_gb": {"type": "number", "description": "Storage amount in GB"},
            "storage_class": {"type": "string", "description": "S3 storage class (STANDARD, IA, GLACIER, etc.)"},
            "region": {"type": "string", "description": "AWS region"}
        },
        "required": ["storage_gb"]
    },
}


def test_7_messages_api_tools_change(client, model_id):
    """Test 7: Messages API — tools change cascading invalidation.
    
    Compares two sub-tests:
      7A: With explicit cache_control markers (like Test 6 but Messages API)
      7B: WITHOUT markers (Simplified Cache) — does auto-cache also cascade-miss?
    """
    log("\n\n" + "█" * 60)
    log("TEST 7: Messages API — tools change + Simplified Cache comparison")
    log("█" * 60)
    log("7A: Explicit cache_control — same logic as Test 6")
    log("7B: No cache_control — tests Simplified Cache auto-caching behavior")

    system_text = SYSTEM_PROMPT

    # Convert Converse API tools to Messages API format
    tools_v1_msg = [converse_tool_to_messages_tool(t) for t in TOOL_DEFINITIONS]
    tools_v2_msg = tools_v1_msg + [EXTRA_TOOL_MESSAGES]

    messages = [{"role": "user", "content": [
        {"type": "text", "text": "What is the cheapest EC2 option for a web server?"}
    ]}]

    # ── 7A: Explicit cache_control ──
    log("\n" + "─" * 40)
    log("7A: Explicit cache_control markers")
    log("─" * 40)
    
    cache_on = {"tools_last": True, "system_last": True}

    m1 = call_messages_api(client, model_id,
                           [{"type": "text", "text": system_text}],
                           TOOL_DEFINITIONS, messages,
                           "7Aa: Original tools (explicit, expect WRITE)",
                           cache_controls=cache_on)
    time.sleep(2)

    m2 = call_messages_api(client, model_id,
                           [{"type": "text", "text": system_text}],
                           TOOL_DEFINITIONS, messages,
                           "7Ab: Same tools (explicit, expect READ)",
                           cache_controls=cache_on)
    time.sleep(2)

    m3 = call_messages_api(client, model_id,
                           [{"type": "text", "text": system_text}],
                           TOOL_DEFINITIONS + [{"toolSpec": {"name": EXTRA_TOOL_MESSAGES["name"],
                                "description": EXTRA_TOOL_MESSAGES["description"],
                                "inputSchema": {"json": EXTRA_TOOL_MESSAGES["input_schema"]}}}],
                           messages,
                           "7Ac: Changed tools (explicit, expect WRITE — cascade)",
                           cache_controls=cache_on)
    time.sleep(2)

    m4 = call_messages_api(client, model_id,
                           [{"type": "text", "text": system_text}],
                           TOOL_DEFINITIONS, messages,
                           "7Ad: Back to original (explicit, expect READ)",
                           cache_controls=cache_on)

    # ── 7B: No cache_control (Simplified Cache) ──
    log("\n" + "─" * 40)
    log("7B: No explicit cache_control (Simplified Cache)")
    log("─" * 40)

    time.sleep(3)

    m5 = call_messages_api(client, model_id,
                           [{"type": "text", "text": system_text}],
                           TOOL_DEFINITIONS, messages,
                           "7Ba: Original tools (no markers, expect auto-WRITE or NONE)")
    time.sleep(2)

    m6 = call_messages_api(client, model_id,
                           [{"type": "text", "text": system_text}],
                           TOOL_DEFINITIONS, messages,
                           "7Bb: Same tools (no markers, expect auto-READ if Simplified Cache)")
    time.sleep(2)

    m7 = call_messages_api(client, model_id,
                           [{"type": "text", "text": system_text}],
                           TOOL_DEFINITIONS + [{"toolSpec": {"name": EXTRA_TOOL_MESSAGES["name"],
                                "description": EXTRA_TOOL_MESSAGES["description"],
                                "inputSchema": {"json": EXTRA_TOOL_MESSAGES["input_schema"]}}}],
                           messages,
                           "7Bc: Changed tools (no markers, expect WRITE if Simplified, or NONE)")
    time.sleep(2)

    m8 = call_messages_api(client, model_id,
                           [{"type": "text", "text": system_text}],
                           TOOL_DEFINITIONS, messages,
                           "7Bd: Back to original (no markers, expect READ if Simplified)")

    return m1, m2, m3, m4, m5, m6, m7, m8


def generate_conversation_turns(num_turns):
    """Generate multi-turn conversation messages.
    Each turn = 1 user block + 1 assistant block = 2 content blocks.
    """
    topics = [
        ("What is Amazon S3?", "Amazon S3 is an object storage service offering scalability, data availability, security, and performance."),
        ("How does S3 pricing work?", "S3 pricing is based on storage used, requests made, data transfer out, and optional features like analytics."),
        ("What are S3 storage classes?", "S3 offers Standard, Intelligent-Tiering, Standard-IA, One Zone-IA, Glacier Instant Retrieval, Glacier Flexible, and Glacier Deep Archive."),
        ("What is Amazon EC2?", "Amazon EC2 provides scalable computing capacity in the AWS cloud, allowing you to launch virtual servers as needed."),
        ("What EC2 instance types are available?", "EC2 offers General Purpose, Compute Optimized, Memory Optimized, Storage Optimized, and Accelerated Computing families."),
        ("How does EC2 Auto Scaling work?", "EC2 Auto Scaling monitors your applications and automatically adjusts capacity to maintain steady, predictable performance."),
        ("What is Amazon RDS?", "Amazon RDS makes it easy to set up, operate, and scale a relational database in the cloud with support for multiple engines."),
        ("What databases does RDS support?", "RDS supports MySQL, PostgreSQL, MariaDB, Oracle, SQL Server, and Amazon Aurora."),
        ("What is Amazon Lambda?", "AWS Lambda lets you run code without provisioning or managing servers, paying only for the compute time consumed."),
        ("What are Lambda triggers?", "Lambda can be triggered by S3 events, API Gateway, DynamoDB Streams, SNS, SQS, CloudWatch Events, and more."),
        ("What is Amazon DynamoDB?", "DynamoDB is a fully managed NoSQL database service that provides fast and predictable performance with seamless scalability."),
        ("How does DynamoDB pricing work?", "DynamoDB pricing is based on read/write capacity units, storage, and optional features like DAX, global tables, and backups."),
    ]
    messages = []
    for i in range(num_turns):
        q, a = topics[i % len(topics)]
        # Add turn number to make each message unique
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": f"[Turn {i+1}] {q}"}]
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": f"[Turn {i+1}] {a}"}]
        })
    return messages


def test_8_block_growth_beyond_20(client, model_id):
    """Test 8: Messages API — cache behavior when conversation grows beyond 20 content blocks.
    
    Each turn = 1 user message + 1 assistant message = 2 content blocks.
    We place cache_control on the last assistant message's content block.
    
    Phases:
      8a: 9 turns (18 blocks) + new user = 19 blocks — under 20, cache write
      8b: Same 9 turns + same user — should cache read
      8c: 10 turns (20 blocks) + new user = 21 blocks — crosses 20 boundary
      8d: Same 10 turns + same user — does previous cache still hit?
      8e: 11 turns (22 blocks) + new user = 23 blocks — further growth
      8f: Same 11 turns + same user — cache behavior?
    """
    log("\n\n" + "█" * 60)
    log("TEST 8: Block growth beyond 20 — cache_control on last assistant")
    log("█" * 60)
    log("Using Messages API with explicit cache_control on last assistant block")
    log("Tracking cache behavior as conversation grows past 20 content blocks")

    system_blocks = [{"type": "text", "text": SYSTEM_PROMPT}]
    new_question = {"role": "user", "content": [
        {"type": "text", "text": "Now summarize everything we discussed."}
    ]}

    results = []

    for num_turns, label in [
        (9, "18 blocks + 1 user = 19 total"),
        (10, "20 blocks + 1 user = 21 total"),
        (11, "22 blocks + 1 user = 23 total"),
    ]:
        history = generate_conversation_turns(num_turns)
        total_blocks = num_turns * 2 + 1  # +1 for the new user question

        # Deep copy to avoid mutation between calls
        import copy

        # Add cache_control to last assistant message's content
        last_assistant_content = history[-1]["content"][-1]
        last_assistant_content["cache_control"] = {"type": "ephemeral"}

        # Also mark system
        system_with_cache = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

        messages = history + [new_question]

        # Call A: First call — should write
        label_a = f"8-{num_turns}a: {label} — first call"
        m_a = call_messages_api(client, model_id, copy.deepcopy(system_with_cache), None,
                                copy.deepcopy(messages), label_a, cache_controls=None)
        time.sleep(2)

        # Call B: Same — should read
        label_b = f"8-{num_turns}b: {label} — same (cache check)"
        m_b = call_messages_api(client, model_id, copy.deepcopy(system_with_cache), None,
                                copy.deepcopy(messages), label_b, cache_controls=None)
        
        results.extend([m_a, m_b])
        time.sleep(3)

    # Summary
    log("\n" + "─" * 40)
    log("Test 8 Summary: Block growth")
    log("─" * 40)
    block_counts = [19, 19, 21, 21, 23, 23]
    labels = ["9t-first", "9t-same", "10t-first", "10t-same", "11t-first", "11t-same"]
    for i, (m, bc, lb) in enumerate(zip(results, block_counts, labels)):
        if m:
            status = "✅ HIT" if m['cacheReadInputTokens'] > 0 else \
                     "📝 WRITE" if m['cacheWriteInputTokens'] > 0 else \
                     "⚪ NONE"
            log(f"  {lb} ({bc} blocks): {status} read={m['cacheReadInputTokens']} write={m['cacheWriteInputTokens']} input={m['inputTokens']}")

    return tuple(results)


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Bedrock Claude Prompt Caching POC")
    parser.add_argument("--model", default="anthropic.claude-sonnet-4-20250514-v1:0",
                       help="Model ID (default: Claude Sonnet 4)")
    parser.add_argument("--region", default="us-east-1",
                       help="AWS region (default: us-east-1)")
    parser.add_argument("--test", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8],
                       help="Run specific test only (1-8)")
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
        7: ("Messages API tools change + Simplified Cache", test_7_messages_api_tools_change),
        8: ("Block growth beyond 20", test_8_block_growth_beyond_20),
    }
    
    run_tests = [args.test] if args.test else [1, 2, 3, 4, 5, 6, 7, 8]
    
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
