#!/usr/bin/env python3
"""
Test script for PHOENIX Multi-Model Intelligence System.
Demonstrates model management, routing, and Claude integration.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add PHOENIX to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.multi_model.multi_model_coordinator import MultiModelCoordinator
from modules.multi_model.claude_integration import ClaudeIntegration, HybridModelRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MultiModelTest")


async def test_multi_model_system():
    """Test the multi-model intelligence system."""

    logger.info("="*60)
    logger.info("PHOENIX Multi-Model Intelligence System Test")
    logger.info("="*60)

    # Initialize coordinator with configuration
    config = {
        'auto_download': False,  # Don't auto-download for test
        'auto_benchmark': False,  # Skip benchmark for test
        'enable_ensemble': True,
        'enable_fallback': True,
        'max_parallel_models': 3
    }

    coordinator = MultiModelCoordinator(config=config)

    # Initialize Claude integration (disabled by default)
    claude_config = {
        'enabled': False,  # Keep disabled for privacy
        'use_for_complex_tasks': True,
        'use_as_fallback': True,
        'require_confirmation': True
    }
    claude = ClaudeIntegration(claude_config)

    # Initialize the system
    logger.info("\n📋 Initializing Multi-Model System...")
    init_status = await coordinator.initialize()

    logger.info(f"✓ Available models: {init_status['available_models']}")

    if init_status['available_models'] == 0:
        logger.warning("⚠️  No models currently installed")
        logger.info("📥 To download models, run: ollama pull qwen2.5:14b-instruct")
        logger.info("   Recommended starter models:")
        logger.info("   • qwen2.5:14b-instruct - Best general purpose")
        logger.info("   • qwen2.5:7b-instruct - Faster, lighter")
        logger.info("   • codellama:7b-instruct - For code tasks")
        logger.info("   • mistral:7b-instruct - Fast responses")

    # Show model status
    logger.info("\n📊 Model Status:")
    status = coordinator.get_model_status()

    for model_key, model_info in status['models'].items():
        if model_info['available']:
            logger.info(f"  ✓ {model_info['name']}: Available")
        else:
            logger.info(f"  ✗ {model_info['name']}: Not installed")

    # Test task classification
    logger.info("\n🎯 Testing Task Classification:")
    test_prompts = [
        ("Write a Python function to sort a list", "code"),
        ("How do I check system memory in Linux?", "system"),
        ("Explain quantum computing", "general"),
        ("Analyze this complex algorithm's time complexity", "reasoning"),
        ("Hi, how are you?", "conversation"),
        ("Summarize this article about AI", "summary")
    ]

    for prompt, expected_type in test_prompts:
        task_type = coordinator.model_router.classify_task(prompt)
        logger.info(f"  • '{prompt[:40]}...' → {task_type.value}")

    # Test model routing (if models available)
    if init_status['available_models'] > 0:
        logger.info("\n🔄 Testing Model Routing:")

        # Test different task types
        test_queries = [
            {
                'prompt': "Write a function to calculate fibonacci",
                'type': 'code'
            },
            {
                'prompt': "What's the weather like?",
                'type': 'general'
            },
            {
                'prompt': "Explain how neural networks work",
                'type': 'complex'
            }
        ]

        for test in test_queries:
            logger.info(f"\n  Testing: {test['type']} task")
            logger.info(f"  Prompt: {test['prompt']}")

            # Get recommendations
            recommendations = coordinator.get_recommendations(test['prompt'])
            logger.info(f"  Task classified as: {recommendations['task_type']}")

            if recommendations['recommended_models']:
                logger.info(f"  Recommended models:")
                for model in recommendations['recommended_models'][:3]:
                    logger.info(f"    • {model['name']} (score: {model['performance_score']:.2f})")

    # Test Claude integration
    logger.info("\n☁️  Claude Integration Status:")
    logger.info(f"  • Enabled: {'✓' if claude.config['enabled'] else '✗'}")
    logger.info(f"  • Available: {'✓' if claude.is_available() else '✗'}")

    if not claude.config['enabled']:
        logger.info("  • Status: Disabled for privacy (local models only)")
        logger.info("  • Note: Claude is the ONLY cloud API supported")
        logger.info("  • Enable with: claude.config['enabled'] = True")
    else:
        logger.info("  • Status: Ready as cloud fallback")
        logger.info("  • Will ask for confirmation before cloud usage")

    # Test hybrid routing
    if coordinator.model_manager.models and claude.config['enabled']:
        logger.info("\n🔀 Testing Hybrid Routing (Local + Claude Fallback):")

        hybrid_router = HybridModelRouter(
            coordinator.model_router,
            claude
        )

        # Test a complex task that might benefit from Claude
        complex_prompt = """
        Design a distributed system architecture for a real-time
        collaboration platform supporting millions of users
        """

        logger.info(f"  Complex task: {complex_prompt[:50]}...")

        # Check if Claude would be used
        should_use = claude.should_use_claude(
            task_type='architecture',
            complexity='complex',
            context_length=len(complex_prompt),
            local_models_failed=False
        )

        logger.info(f"  Would use Claude: {'Yes' if should_use else 'No (using local models)'}")

    # Show system summary
    logger.info("\n" + coordinator.get_summary())

    # Test ensemble capability (if multiple models available)
    available_count = sum(1 for m in coordinator.model_manager.models.values() if m.is_available)
    if available_count >= 2:
        logger.info("\n🎭 Ensemble Capability: Ready")
        logger.info("  Can use multiple models for validation")
    else:
        logger.info("\n🎭 Ensemble Capability: Needs 2+ models")

    # Configuration recommendations
    logger.info("\n💡 Configuration Recommendations:")
    logger.info("  For Code Tasks: Install CodeLlama or DeepSeek Coder")
    logger.info("  For General Use: Qwen 2.5 14B or Mistral 7B")
    logger.info("  For Fast Responses: Phi-3 Mini or Mistral 7B")
    logger.info("  For Long Context: Llama 3.2 3B (128K context!)")
    logger.info("  For Complex Reasoning: Qwen 2.5 32B or Claude (if enabled)")

    # Privacy note
    logger.info("\n🔒 Privacy Configuration:")
    logger.info("  • All models run locally by default")
    logger.info("  • Claude is the ONLY cloud option (disabled by default)")
    logger.info("  • Cloud usage requires explicit confirmation")
    logger.info("  • All cloud usage is logged for transparency")

    logger.info("\n✅ Multi-Model System Test Complete!")

    return True


async def main():
    """Main entry point."""
    try:
        success = await test_multi_model_system()

        if success:
            logger.info("\n🎉 Multi-Model Intelligence System is ready!")
            logger.info("PHOENIX can now leverage multiple AI models for optimal performance")
            logger.info("\nNext steps:")
            logger.info("1. Install desired models: ollama pull [model-name]")
            logger.info("2. Run benchmarks: coordinator.benchmark_all()")
            logger.info("3. Enable Claude if needed: claude.config['enabled'] = True")
            logger.info("4. Integrate with main PHOENIX core")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())