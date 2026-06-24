#!/usr/bin/env python3
"""
Mixture-of-Agents Tool Module

This module implements the Mixture-of-Agents (MoA) methodology that leverages
the collective strengths of multiple LLMs through a layered architecture to
achieve state-of-the-art performance on complex reasoning tasks.

Based on the research paper: "Mixture-of-Agents Enhances Large Language Model Capabilities"
by Junlin Wang et al. (arXiv:2406.04692v1)

Architecture (2-layer, 2-model):
1. Reference model: DeepSeek-V4-Flash (OpenRouter) — fast, diverse perspective
2. Aggregator model: Main model via provider router (default: GLM-5.2 via zai) — synthesis

Total API calls: 2 (1 reference + 1 aggregator).  Cost-optimized for regular use.

Configuration:
    To customize the MoA setup, modify the configuration constants at the top of this file:
    - REFERENCE_MODEL: Single reference model for generating an initial response
    - AGGREGATOR_PROVIDER: Provider for the aggregator (default: "zai" for GLM-5.2)
    - AGGREGATOR_MODEL: Model slug for the aggregator (default: "glm-5-turbo")
    - REFERENCE_TEMPERATURE/AGGREGATOR_TEMPERATURE: Sampling temperatures
"""

import json
import logging
import os
import asyncio
import datetime
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from agent.auxiliary_client import resolve_provider_client, extract_content_or_reasoning
from tools.debug_helpers import DebugSession
import sys

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

# Layer 1: 主模型 GLM-5.2（智谱官方 API）— 先给出初始回答
# zai key 由 resolve_provider_client("zai") 从 config.yaml 读取
REFERENCE_PROVIDER = "zai"
REFERENCE_MODEL = "glm-5-turbo"

# Layer 2: 交叉模型 DeepSeek-V4-Flash（官方 API）— 交叉验证/综合优化
AGGREGATOR_MODEL = "deepseek-v4-flash"
AGGREGATOR_BASE_URL = "https://api.deepseek.com/v1"
AGGREGATOR_API_KEY_ENV = "DEEPSEEK_API_KEY"

# Temperature settings
REFERENCE_TEMPERATURE = 0.1   # 主模型
AGGREGATOR_TEMPERATURE = 0.1  # 交叉模型

# System prompt for the aggregator model (from the research paper)
AGGREGATOR_SYSTEM_PROMPT = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

_debug = DebugSession("moa_tools", env_var="MOA_TOOLS_DEBUG")


def _construct_aggregator_prompt(system_prompt: str, responses: List[str]) -> str:
    """
    Construct the final system prompt for the aggregator including all model responses.
    
    Args:
        system_prompt (str): Base system prompt for aggregation
        responses (List[str]): List of responses from reference models
        
    Returns:
        str: Complete system prompt with enumerated responses
    """
    response_text = "\n".join([f"{i+1}. {response}" for i, response in enumerate(responses)])
    return f"{system_prompt}\n\n{response_text}"


async def _run_reference_model(
    user_prompt: str,
    max_retries: int = 3
) -> str:
    """Layer 1: 主模型 GLM-5.2 先给出初始回答。

    优先用 resolve_provider_client("zai") 走 Hermes 统一路由，
    失败则 fallback 到 AsyncOpenAI + GLM_API_KEY。
    """
    # 1. 尝试 Hermes 统一路由
    try:
        client, resolved_model = resolve_provider_client("zai", model=REFERENCE_MODEL, async_mode=True)
        if client is not None:
            logger.info("Layer 1: GLM via zai provider: %s", resolved_model or REFERENCE_MODEL)
            for attempt in range(max_retries):
                try:
                    api_params = {
                        "model": resolved_model or REFERENCE_MODEL,
                        "messages": [{"role": "user", "content": user_prompt}],
                        "temperature": REFERENCE_TEMPERATURE,
                    }
                    response = await client.chat.completions.create(**api_params)
                    content = extract_content_or_reasoning(response)
                    if content:
                        logger.info("Layer 1 complete: GLM responded (%s chars)", len(content))
                        return content
                    logger.warning("Layer 1: GLM returned empty (attempt %s/%s)", attempt + 1, max_retries)
                except Exception as e:
                    logger.warning("Layer 1: GLM error (attempt %s/%s): %s", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
    except Exception as e:
        logger.warning("Layer 1: zai provider unavailable: %s, trying fallback", e)

    # 2. Fallback: AsyncOpenAI + GLM_API_KEY
    api_key = os.getenv("GLM_API_KEY", "")
    if not api_key:
        raise RuntimeError("zai provider unavailable and GLM_API_KEY not set")
    logger.info("Layer 1 fallback: AsyncOpenAI + GLM_API_KEY")
    client = AsyncOpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4")
    for attempt in range(max_retries):
        try:
            api_params = {
                "model": REFERENCE_MODEL,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": REFERENCE_TEMPERATURE,
            }
            response = await client.chat.completions.create(**api_params)
            content = extract_content_or_reasoning(response)
            if content:
                logger.info("Layer 1 complete via fallback (%s chars)", len(content))
                return content
        except Exception as e:
            logger.warning("Layer 1 fallback error (attempt %s/%s): %s", attempt + 1, max_retries, e)
        if attempt < max_retries - 1:
            await asyncio.sleep(2)
    raise RuntimeError(f"Layer 1: GLM {REFERENCE_MODEL} failed after {max_retries} attempts")


async def _run_aggregator_model(
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 2
) -> str:
    """Layer 2: 交叉模型 DeepSeek-V4-Flash 验证/综合优化。

    DeepSeek 走 OpenAI 兼容协议，用 DEEPSEEK_API_KEY。
    """
    api_key = os.getenv(AGGREGATOR_API_KEY_ENV, "")
    if not api_key:
        raise RuntimeError(f"{AGGREGATOR_API_KEY_ENV} not set — export it or set in .env")
    client = AsyncOpenAI(api_key=api_key, base_url=AGGREGATOR_BASE_URL)

    for attempt in range(max_retries):
        try:
            logger.info("Layer 2: DeepSeek %s (attempt %s/%s)", AGGREGATOR_MODEL, attempt + 1, max_retries)
            api_params = {
                "model": AGGREGATOR_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 32000,
                "temperature": AGGREGATOR_TEMPERATURE,
            }
            response = await client.chat.completions.create(**api_params)
            content = extract_content_or_reasoning(response)
            if content:
                logger.info("Layer 2 complete: DeepSeek aggregation (%s chars)", len(content))
                return content
            logger.warning("Layer 2: DeepSeek returned empty (attempt %s/%s)", attempt + 1, max_retries)
        except Exception as e:
            logger.warning("Layer 2: DeepSeek error (attempt %s/%s): %s", attempt + 1, max_retries, e)
        if attempt < max_retries - 1:
            await asyncio.sleep(min(2 ** (attempt + 1), 30))
    raise RuntimeError(f"Layer 2: DeepSeek {AGGREGATOR_MODEL} failed after {max_retries} attempts")


async def mixture_of_agents_tool(
    user_prompt: str,
) -> str:
    """Mixture-of-Agents: 2-layer, 2-model.

    Layer 1: GLM-5.2 (主模型) 给出初始回答。
    Layer 2: DeepSeek-V4-Flash (交叉模型) 验证并综合优化。

    Total API calls: 2.  Use for genuinely hard problems.
    """
    start_time = datetime.datetime.now()

    try:
        logger.info("Starting MoA (2-model): L1=%s, L2=%s", REFERENCE_MODEL, AGGREGATOR_MODEL)

        # Validate requirements
        try:
            ref_client, _ = resolve_provider_client("zai", model=REFERENCE_MODEL, async_mode=True)
            if ref_client is None:
                raise ValueError("zai provider not configured")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"zai provider unavailable: {e}")
        if not os.getenv(AGGREGATOR_API_KEY_ENV):
            raise ValueError(f"{AGGREGATOR_API_KEY_ENV} not set")

        # Layer 1: 主模型 GLM
        logger.info("Layer 1: Querying GLM...")
        ref_response = await _run_reference_model(user_prompt)
        logger.info("Layer 1 complete (%s chars)", len(ref_response))

        # Layer 2: 交叉模型 DeepSeek
        logger.info("Layer 2: Cross-validation with DeepSeek...")
        agg_prompt = _construct_aggregator_prompt(AGGREGATOR_SYSTEM_PROMPT, [ref_response])
        final_response = await _run_aggregator_model(agg_prompt, user_prompt)

        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.info("MoA complete in %.2fs", processing_time)

        result = {
            "success": True,
            "response": final_response,
            "models_used": {
                "layer1_model": REFERENCE_MODEL,
                "layer2_model": AGGREGATOR_MODEL,
            },
            "processing_time": round(processing_time, 2),
        }
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error in MoA processing: {e}"
        logger.error("%s", error_msg, exc_info=True)
        result = {
            "success": False,
            "response": "MoA processing failed. Please try again or use a single model for this query.",
            "models_used": {
                "layer1_model": REFERENCE_MODEL,
                "layer2_model": AGGREGATOR_MODEL,
            },
            "error": error_msg,
        }
        return json.dumps(result, indent=2, ensure_ascii=False)


def check_moa_requirements() -> bool:
    """MoA 需要 zai provider 可用 + DEEPSEEK_API_KEY。"""
    try:
        client, _ = resolve_provider_client("zai", async_mode=True)
        if client is None:
            return False
    except Exception as e:  # noqa: availability check — False is correct fallback
        logger.debug("zai provider check failed: %s", e)
        return False
    return bool(os.getenv(AGGREGATOR_API_KEY_ENV))


def get_moa_configuration() -> Dict[str, Any]:
    """当前 MoA 配置。"""
    return {
        "layer1_model": REFERENCE_MODEL,
        "layer1_provider": REFERENCE_PROVIDER,
        "layer2_model": AGGREGATOR_MODEL,
        "layer2_base_url": AGGREGATOR_BASE_URL,
        "reference_temperature": REFERENCE_TEMPERATURE,
        "aggregator_temperature": AGGREGATOR_TEMPERATURE,
        "total_api_calls": 2,
    }


if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🤖 Mixture-of-Agents Tool Module")
    print("=" * 50)
    
    # Check if API key is available
    api_available = check_openrouter_api_key()
    
    if not api_available:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("Please set your API key: export OPENROUTER_API_KEY='your-key-here'")
        print("Get API key at: https://openrouter.ai/")
        sys.exit(1)
    else:
        print("✅ OpenRouter API key found")
    
    print("🛠️  MoA tools ready for use!")
    
    # Show current configuration
    config = get_moa_configuration()
    print(f"\n  Layer 1: {config['layer1_model']} ({config['layer1_provider']})")
    print(f"  Layer 2: {config['layer2_model']} ({config['layer2_base_url']})")
    print(f"  API calls: {config['total_api_calls']}")
    print(f"  Reference temp: {config['reference_temperature']}, Aggregator temp: {config['aggregator_temperature']}")

    print("\nUsage:")
    print("  from tools.mixture_of_agents_tool import mixture_of_agents_tool")
    print("  import asyncio")
    print("  asyncio.run(mixture_of_agents_tool(user_prompt='...'))")
    print("")
    print("\nDebug mode:")
    print("  # Enable debug logging")
    print("  export MOA_TOOLS_DEBUG=true")
    print("  # Debug logs capture all MoA processing steps and metrics")
    print("  # Logs saved to: ./logs/moa_tools_debug_UUID.json")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

MOA_SCHEMA = {
    "name": "mixture_of_agents",
    "description": "Route a hard problem through 2 LLMs: Layer 1 GLM-5.2 gives an initial answer, Layer 2 DeepSeek-V4-Flash cross-validates and refines. 2 API calls, no OpenRouter needed. Best for: complex math, algorithms, multi-step reasoning, problems needing a second perspective.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_prompt": {
                "type": "string",
                "description": "The complex query or problem to solve. Should be a challenging problem that benefits from collaborative reasoning."
            }
        },
        "required": ["user_prompt"]
    }
}

registry.register(
    name="mixture_of_agents",
    toolset="moa",
    schema=MOA_SCHEMA,
    handler=lambda args, **kw: mixture_of_agents_tool(user_prompt=args.get("user_prompt", "")),
    check_fn=check_moa_requirements,
    requires_env=["DEEPSEEK_API_KEY"],
    is_async=True,
    emoji="🧠",
)
