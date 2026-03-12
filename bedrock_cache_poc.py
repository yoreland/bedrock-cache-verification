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


def generate_conversation_turns(num_turns, blocks_per_assistant=1):
    """Generate multi-turn conversation messages.
    
    Args:
        num_turns: Number of conversation turns.
        blocks_per_assistant: Number of text blocks in each assistant message.
            1 = standard (1 user block + 1 assistant block = 2 blocks per turn)
            3 = multi-block (1 user block + 3 assistant blocks = 4 blocks per turn)
    """
    topics = [
        ("What is Amazon S3?",
         ["Amazon S3 is an object storage service offering scalability, data availability, security, and performance.",
          "S3 stores data as objects within buckets, with virtually unlimited storage capacity.",
          "It supports lifecycle policies, versioning, encryption, and cross-region replication."]),
        ("How does S3 pricing work?",
         ["S3 pricing is based on storage used, requests made, data transfer out, and optional features.",
          "Storage classes have different price tiers: Standard is most expensive, Glacier Deep Archive is cheapest.",
          "Data transfer IN is free, but transfer OUT and API requests incur charges."]),
        ("What are S3 storage classes?",
         ["S3 offers Standard, Intelligent-Tiering, Standard-IA, One Zone-IA, and Glacier tiers.",
          "Intelligent-Tiering automatically moves objects between access tiers based on usage patterns.",
          "Glacier classes are for archival with retrieval times from minutes to hours."]),
        ("What is Amazon EC2?",
         ["Amazon EC2 provides scalable computing capacity in the AWS cloud.",
          "You can launch virtual servers (instances) with various CPU, memory, and storage configurations.",
          "EC2 supports multiple pricing models: On-Demand, Reserved, Spot, and Savings Plans."]),
        ("What EC2 instance types are available?",
         ["EC2 offers General Purpose (M/T), Compute Optimized (C), Memory Optimized (R/X) families.",
          "Storage Optimized (I/D) instances provide high sequential read/write for large datasets.",
          "Accelerated Computing (P/G) instances use hardware accelerators like GPUs."]),
        ("How does EC2 Auto Scaling work?",
         ["EC2 Auto Scaling monitors applications and automatically adjusts capacity for performance.",
          "You define scaling policies based on metrics like CPU utilization or request count.",
          "It can scale across multiple AZs for high availability and cost optimization."]),
        ("What is Amazon RDS?",
         ["Amazon RDS simplifies setup, operation, and scaling of relational databases in the cloud.",
          "It automates routine tasks like hardware provisioning, database setup, and patching.",
          "RDS provides automated backups, database snapshots, and multi-AZ deployments."]),
        ("What databases does RDS support?",
         ["RDS supports MySQL, PostgreSQL, MariaDB, Oracle, SQL Server, and Amazon Aurora.",
          "Aurora is Amazon's cloud-native database with up to 5x MySQL and 3x PostgreSQL throughput.",
          "Each engine has specific version support and feature availability."]),
        ("What is AWS Lambda?",
         ["AWS Lambda lets you run code without provisioning or managing servers.",
          "You pay only for the compute time consumed, billed in 1ms increments.",
          "Lambda automatically scales from a few requests per day to thousands per second."]),
        ("What are Lambda triggers?",
         ["Lambda can be triggered by S3 events, API Gateway, DynamoDB Streams, and more.",
          "Event source mappings allow Lambda to poll services like SQS, Kinesis, and Kafka.",
          "You can also invoke Lambda directly via the SDK or create custom event sources."]),
        ("What is Amazon DynamoDB?",
         ["DynamoDB is a fully managed NoSQL database providing fast, predictable performance.",
          "It supports key-value and document data models with single-digit millisecond latency.",
          "DynamoDB offers built-in security, backup/restore, and in-memory caching."]),
        ("How does DynamoDB pricing work?",
         ["DynamoDB pricing is based on read/write capacity units and storage consumed.",
          "On-Demand mode charges per request; Provisioned mode charges for allocated capacity.",
          "Optional features like DAX, global tables, and backups have separate pricing."]),
    ]
    messages = []
    for i in range(num_turns):
        q, answers = topics[i % len(topics)]
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": f"[Turn {i+1}] {q}"}]
        })
        # Use requested number of blocks for assistant
        assistant_blocks = []
        for b in range(blocks_per_assistant):
            text = answers[b % len(answers)] if b < len(answers) else answers[-1]
            assistant_blocks.append({"type": "text", "text": f"[Turn {i+1}.{b+1}] {text}"})
        messages.append({
            "role": "assistant",
            "content": assistant_blocks
        })
    return messages


def count_content_blocks(messages):
    """Count total content blocks across all messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            total += len(content)
        else:
            total += 1
    return total


def test_8_simplified_cache_20_block_boundary(client, model_id):
    """Test 8: Simplified Cache — behavior at the 20 content block boundary.
    
    Per AWS docs, simplified cache management places ONE cache checkpoint
    and the system looks back ~20 content blocks from that point.
    
    This test uses multi-block assistant messages (3 text blocks each) to quickly
    exceed 20 blocks, with only ONE cache_control marker on the last assistant block.
    
    Question: When total blocks > 20, do early blocks (outside the ~20 lookback window)
    lose cache coverage?
    
    Phase A: 4 turns (4 user + 4*3 assistant = 16 blocks) + 1 user = 17 blocks
    Phase B: 5 turns (5 user + 5*3 assistant = 20 blocks) + 1 user = 21 blocks  
    Phase C: 7 turns (7 user + 7*3 assistant = 28 blocks) + 1 user = 29 blocks
    Phase D: 10 turns (10 user + 10*3 assistant = 40 blocks) + 1 user = 41 blocks
    
    Only ONE cache_control on the very last assistant block (simplified mode).
    """
    log("\n\n" + "█" * 60)
    log("TEST 8: Simplified Cache — 20 content block boundary")
    log("█" * 60)
    log("ONE cache_control on last assistant block only (simplified mode)")
    log("Assistant messages have 3 text blocks each to grow block count fast")
    log("Tracking: does lookback window limit cache coverage?")

    import copy

    new_question = {"role": "user", "content": [
        {"type": "text", "text": "Now summarize everything we discussed."}
    ]}

    test_configs = [
        (4, "4 turns"),
        (5, "5 turns"),
        (7, "7 turns"),
        (10, "10 turns"),
    ]

    results = []
    result_labels = []

    for num_turns, turn_label in test_configs:
        history = generate_conversation_turns(num_turns, blocks_per_assistant=3)
        messages = history + [new_question]
        total_blocks = count_content_blocks(messages)

        # Only ONE cache_control on last assistant block (simplified mode)
        # Find last assistant message, mark its last content block
        for msg in reversed(history):
            if msg["role"] == "assistant":
                msg["content"][-1]["cache_control"] = {"type": "ephemeral"}
                break

        log(f"\n  {turn_label}: {total_blocks} content blocks (user: {num_turns}, "
            f"assistant: {num_turns}×3={num_turns*3}, +1 final user)")

        # Call A: First
        label_a = f"8-{turn_label}: {total_blocks} blocks — first call"
        m_a = call_messages_api(client, model_id, 
                                [{"type": "text", "text": SYSTEM_PROMPT}],
                                None, copy.deepcopy(messages), label_a)
        time.sleep(2)

        # Call B: Same — cache check
        label_b = f"8-{turn_label}: {total_blocks} blocks — same (cache check)"
        m_b = call_messages_api(client, model_id,
                                [{"type": "text", "text": SYSTEM_PROMPT}],
                                None, copy.deepcopy(messages), label_b)

        results.extend([m_a, m_b])
        result_labels.extend([
            f"{turn_label}-first ({total_blocks} blocks)",
            f"{turn_label}-same  ({total_blocks} blocks)",
        ])
        time.sleep(3)

    # Summary
    log("\n" + "─" * 50)
    log("Test 8 Summary: Simplified cache 20-block boundary")
    log("─" * 50)
    for lb, m in zip(result_labels, results):
        if m:
            status = "✅ HIT" if m['cacheReadInputTokens'] > 0 else \
                     "📝 WRITE" if m['cacheWriteInputTokens'] > 0 else \
                     "⚪ NONE"
            log(f"  {lb}: {status} read={m['cacheReadInputTokens']} "
                f"write={m['cacheWriteInputTokens']} input={m['inputTokens']}")

    return tuple(results)


def test_9_lookback_window_verification(client, model_id):
    """Test 9: Clean verification of simplified cache 20-block lookback window.
    
    Uses UUID-stamped content to guarantee no cache residue from prior runs.
    
    Two sub-tests:
      9A: Simplified mode (1 cache_control on last assistant only)
          - Establish 30-block cache
          - Modify early block (#1) vs late block (#25) — does lookback matter?
          - Modify system / add tools — full miss?
          
      9B: Manual mode (cache_control on system + last assistant = 2 checkpoints)
          - Same modifications — does independent checkpoint evaluation help?
    """
    import copy
    import uuid

    run_id = uuid.uuid4().hex[:8]  # unique per run to avoid cache residue
    log("\n\n" + "█" * 60)
    log(f"TEST 9: Clean lookback verification (run_id={run_id})")
    log("█" * 60)
    log("All content stamped with unique run_id to avoid cache residue")

    # Unique system prompt for this run
    system_text = f"{SYSTEM_PROMPT}\n[run_id={run_id}]"

    def build_conversation(num_turns=15, modify_block=None, extra_system_text=None):
        """Build a unique conversation. 15 turns = 30 blocks + 1 final user = 31 blocks."""
        msgs = []
        block_idx = 0
        for i in range(num_turns):
            user_text = f"[{run_id}][Turn {i+1}] Tell me about AWS service #{i+1}."
            asst_text = (f"[{run_id}][Turn {i+1}] Service {i+1} provides HA, security, "
                         f"scalability, multi-region, and integrates with other AWS services.")

            if modify_block is not None and block_idx == modify_block:
                user_text = f"[{run_id}][Turn {i+1}] MODIFIED QUESTION about service #{i+1}."
            msgs.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
            block_idx += 1

            if modify_block is not None and block_idx == modify_block:
                asst_text = f"[{run_id}][Turn {i+1}] MODIFIED ANSWER about service #{i+1}."
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": asst_text}]})
            block_idx += 1

        msgs.append({"role": "user", "content": [
            {"type": "text", "text": f"[{run_id}] Summarize all services."}
        ]})

        sys = [{"type": "text", "text": extra_system_text or system_text}]
        return sys, msgs

    results = []
    labels = []

    def run_call(sys, tools, msgs, label):
        m = call_messages_api(client, model_id, copy.deepcopy(sys), tools,
                              copy.deepcopy(msgs), label)
        results.append(m)
        labels.append(label)
        time.sleep(2)
        return m

    # ═══════════════════════════════════════════
    # 9A: Simplified mode — 1 checkpoint only
    # ═══════════════════════════════════════════
    log("\n" + "─" * 50)
    log("9A: SIMPLIFIED MODE (1 cache_control on last assistant)")
    log("─" * 50)

    sys_orig, msgs_orig = build_conversation()
    # Only 1 cache_control: last assistant block
    msgs_orig[-2]["content"][-1]["cache_control"] = {"type": "ephemeral"}

    run_call(sys_orig, None, msgs_orig, "9Aa: establish cache (31 blocks)")
    run_call(sys_orig, None, msgs_orig, "9Ab: same — verify warm")

    # Modify early block (#1 = Turn 1 assistant)
    sys_early, msgs_early = build_conversation(modify_block=1)
    msgs_early[-2]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    run_call(sys_early, None, msgs_early, "9Ac: modify block #1 (early)")

    # Modify late block (#25 = Turn 13 assistant)
    sys_late, msgs_late = build_conversation(modify_block=25)
    msgs_late[-2]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    run_call(sys_late, None, msgs_late, "9Ad: modify block #25 (late)")

    # Modify system prompt
    sys_mod, msgs_sysmod = build_conversation(extra_system_text=system_text + " Always mention pricing.")
    msgs_sysmod[-2]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    run_call(sys_mod, None, msgs_sysmod, "9Ae: modified system")

    # Add tools (original system + messages)
    run_call(sys_orig, TOOL_DEFINITIONS, msgs_orig, "9Af: added tools")

    # Extend: add 1 more turn (33 blocks)
    sys_ext, msgs_ext = build_conversation(num_turns=16)
    msgs_ext[-2]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    run_call(sys_ext, None, msgs_ext, "9Ag: extended to 33 blocks")

    # Back to original
    run_call(sys_orig, None, msgs_orig, "9Ah: back to original")

    # ═══════════════════════════════════════════
    # 9B: Manual mode — 2 checkpoints
    # ═══════════════════════════════════════════
    log("\n" + "─" * 50)
    log("9B: MANUAL MODE (cache_control on system + last assistant)")
    log("─" * 50)

    run_id_b = uuid.uuid4().hex[:8]
    system_text_b = f"{SYSTEM_PROMPT}\n[run_id={run_id_b}]"

    def build_conv_b(num_turns=15, modify_block=None, extra_system_text=None):
        msgs = []
        block_idx = 0
        for i in range(num_turns):
            user_text = f"[{run_id_b}][Turn {i+1}] Tell me about AWS service #{i+1}."
            asst_text = (f"[{run_id_b}][Turn {i+1}] Service {i+1} provides HA, security, "
                         f"scalability, multi-region, and integrates with other AWS services.")
            if modify_block is not None and block_idx == modify_block:
                user_text = f"[{run_id_b}][Turn {i+1}] MODIFIED QUESTION about service #{i+1}."
            msgs.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
            block_idx += 1
            if modify_block is not None and block_idx == modify_block:
                asst_text = f"[{run_id_b}][Turn {i+1}] MODIFIED ANSWER about service #{i+1}."
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": asst_text}]})
            block_idx += 1
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": f"[{run_id_b}] Summarize all services."}
        ]})
        sys = [{"type": "text", "text": extra_system_text or system_text_b}]
        return sys, msgs

    # 2 checkpoints: system + last assistant
    def add_manual_markers(sys, msgs):
        sys[0]["cache_control"] = {"type": "ephemeral"}
        msgs[-2]["content"][-1]["cache_control"] = {"type": "ephemeral"}

    sys_b_orig, msgs_b_orig = build_conv_b()
    add_manual_markers(sys_b_orig, msgs_b_orig)
    run_call(sys_b_orig, None, msgs_b_orig, "9Ba: establish cache (2 checkpoints)")
    run_call(sys_b_orig, None, msgs_b_orig, "9Bb: same — verify warm")

    # Modify early block
    sys_b_early, msgs_b_early = build_conv_b(modify_block=1)
    add_manual_markers(sys_b_early, msgs_b_early)
    run_call(sys_b_early, None, msgs_b_early, "9Bc: modify block #1 (early)")

    # Modify late block
    sys_b_late, msgs_b_late = build_conv_b(modify_block=25)
    add_manual_markers(sys_b_late, msgs_b_late)
    run_call(sys_b_late, None, msgs_b_late, "9Bd: modify block #25 (late)")

    # Modify system
    sys_b_mod, msgs_b_sysmod = build_conv_b(extra_system_text=system_text_b + " Always mention pricing.")
    add_manual_markers(sys_b_mod, msgs_b_sysmod)
    run_call(sys_b_mod, None, msgs_b_sysmod, "9Be: modified system (2 checkpoints)")

    # Back to original
    run_call(sys_b_orig, None, msgs_b_orig, "9Bf: back to original")

    # Summary
    log("\n" + "─" * 50)
    log(f"Test 9 Summary (run_id A={run_id}, B={run_id_b})")
    log("─" * 50)
    for lb, m in zip(labels, results):
        if m:
            status = "✅ HIT" if m['cacheReadInputTokens'] > 0 else \
                     "📝 WRITE" if m['cacheWriteInputTokens'] > 0 else \
                     "⚪ NONE"
            log(f"  {lb}: {status} read={m['cacheReadInputTokens']} "
                f"write={m['cacheWriteInputTokens']} input={m['inputTokens']}")

    return tuple(results)


def test_10_sliding_window_cache(client, model_id):
    """Test 10: Sliding window cache — drop early blocks, keep recent ones.
    
    Idea: maintain a ~20 block conversation window. As new turns arrive,
    drop the oldest turns. Use 2 checkpoints (system + last assistant).
    
    Question: when we drop early blocks, does the remaining suffix still
    hit cache from the previous version that included those blocks?
    
    Scenario (all unique content via UUID):
      10a: 10 turns (20 msg blocks) + system checkpoint + last assistant checkpoint → WRITE
      10b: Same → HIT (verify warm)
      10c: Drop first 2 turns, keep turns 3-10 (16 blocks) → HIT or MISS?
      10d: Same as 10c → HIT (verify)
      10e: Drop first 2, add 2 new turns at end (turns 3-12, 20 blocks) → partial HIT?
      10f: Same as 10e → HIT (verify)
      10g: Slide again: turns 5-14 (20 blocks) → partial HIT?
      10h: Same as 10g → HIT (verify)
    """
    import copy
    import uuid

    run_id = uuid.uuid4().hex[:8]
    log("\n\n" + "█" * 60)
    log(f"TEST 10: Sliding window cache (run_id={run_id})")
    log("█" * 60)
    log("2 checkpoints: system + last assistant")
    log("Sliding window: drop old turns, add new ones, check cache behavior")

    system_text = f"{SYSTEM_PROMPT}\n[run_id={run_id}]"

    def build_windowed_conversation(start_turn, end_turn):
        """Build conversation from turn start_turn to end_turn (inclusive).
        Each turn = 1 user + 1 assistant = 2 blocks.
        """
        msgs = []
        for i in range(start_turn, end_turn + 1):
            msgs.append({"role": "user", "content": [
                {"type": "text", "text": f"[{run_id}][Turn {i}] Question about AWS service #{i}."}
            ]})
            msgs.append({"role": "assistant", "content": [
                {"type": "text", "text": (f"[{run_id}][Turn {i}] Service {i} provides HA, security, "
                    f"scalability, multi-region support, and integrates with other AWS services.")}
            ]})
        # Final user question
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": f"[{run_id}] What should I use next?"}
        ]})

        # 2 checkpoints: system + last assistant
        sys = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
        msgs[-2]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        total_blocks = count_content_blocks(msgs)
        return sys, msgs, total_blocks

    results = []
    labels = []

    def run_call(sys, msgs, label, total_blocks=None):
        extra = f" ({total_blocks} blocks)" if total_blocks else ""
        m = call_messages_api(client, model_id, copy.deepcopy(sys), None,
                              copy.deepcopy(msgs), f"{label}{extra}")
        results.append(m)
        labels.append(f"{label}{extra}")
        time.sleep(2)
        return m

    # 10a: Window = turns 1-10 (20 message blocks)
    sys_a, msgs_a, blocks_a = build_windowed_conversation(1, 10)
    run_call(sys_a, msgs_a, "10a: turns 1-10", blocks_a)
    run_call(sys_a, msgs_a, "10b: same (verify)", blocks_a)

    # 10c: Drop turns 1-2, keep turns 3-10 (16 message blocks)
    sys_c, msgs_c, blocks_c = build_windowed_conversation(3, 10)
    run_call(sys_c, msgs_c, "10c: turns 3-10 (dropped 1-2)", blocks_c)
    run_call(sys_c, msgs_c, "10d: same (verify)", blocks_c)

    # 10e: Slide window: turns 3-12 (20 message blocks, added 11-12)
    sys_e, msgs_e, blocks_e = build_windowed_conversation(3, 12)
    run_call(sys_e, msgs_e, "10e: turns 3-12 (slide +2)", blocks_e)
    run_call(sys_e, msgs_e, "10f: same (verify)", blocks_e)

    # 10g: Slide again: turns 5-14 (20 message blocks)
    sys_g, msgs_g, blocks_g = build_windowed_conversation(5, 14)
    run_call(sys_g, msgs_g, "10g: turns 5-14 (slide +2)", blocks_g)
    run_call(sys_g, msgs_g, "10h: same (verify)", blocks_g)

    # Summary
    log("\n" + "─" * 55)
    log(f"Test 10 Summary: Sliding window (run_id={run_id})")
    log("─" * 55)
    for lb, m in zip(labels, results):
        if m:
            status = "✅ HIT" if m['cacheReadInputTokens'] > 0 else \
                     "📝 WRITE" if m['cacheWriteInputTokens'] > 0 else \
                     "⚪ NONE"
            log(f"  {lb}: {status} read={m['cacheReadInputTokens']} "
                f"write={m['cacheWriteInputTokens']}")

    log("\n  KEY INSIGHT:")
    if results[2] and results[2]['cacheReadInputTokens'] > 0:
        log(f"  10c (dropped early turns): PARTIAL HIT read={results[2]['cacheReadInputTokens']}"
            f" → system checkpoint survived! Sliding window viable.")
    else:
        log("  10c (dropped early turns): FULL MISS → dropping early turns = full re-cache")

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
    parser.add_argument("--test", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       help="Run specific test only (1-10)")
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
        8: ("Simplified cache 20-block boundary", test_8_simplified_cache_20_block_boundary),
        9: ("Lookback window verification", test_9_lookback_window_verification),
        10: ("Sliding window cache", test_10_sliding_window_cache),
    }
    
    run_tests = [args.test] if args.test else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
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
