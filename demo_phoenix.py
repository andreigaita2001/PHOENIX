#!/usr/bin/env python3
"""
Interactive demonstration of PHOENIX's enhanced capabilities.
This demonstrates conversation persistence, pattern learning, and module creation.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.phoenix_core import PhoenixCore

async def demo_phoenix():
    """Demonstrate PHOENIX's key features."""
    print("""
    ╔════════════════════════════════════════════╗
    ║         🔥 PHOENIX AI SYSTEM 🔥            ║
    ║     Deep Learning & Self-Improvement       ║
    ║                                            ║
    ║       Demonstrating Key Features           ║
    ╚════════════════════════════════════════════╝
    """)

    # Initialize PHOENIX
    phoenix = PhoenixCore()
    await phoenix.initialize_modules()

    print("\n✨ PHOENIX is now fully operational with:")
    print("  • Conversation persistence across sessions")
    print("  • Pattern recognition and habit learning")
    print("  • Autonomous module creation capability")
    print("  • Multi-model intelligence system")
    print("  • Self-modification framework")

    demos = [
        ("🧠 Testing Memory", "Remember that I love Python programming"),
        ("📊 Pattern Learning", "check system status"),
        ("🔍 Memory Recall", "What do you know about me?"),
        ("🤖 Self-Understanding", "Explain your capabilities and modules"),
        ("🛠️ Module Analysis", "Can you create a new module for scheduling tasks?")
    ]

    for title, command in demos:
        print(f"\n{title}")
        print("=" * 60)
        print(f"User: {command}")
        response = await phoenix.process_command(command)
        print(f"PHOENIX: {response[:500]}...")
        await asyncio.sleep(1)

    # Show statistics
    print("\n📈 System Statistics")
    print("=" * 60)

    if 'memory' in phoenix.modules:
        stats = phoenix.modules['memory'].get_stats()
        print(f"  • Total conversations: {stats['total_conversations']}")
        print(f"  • Knowledge facts: {stats['total_facts']}")
        print(f"  • Vector memories: {stats.get('vector_memories', 0)}")

    if 'pattern_engine' in phoenix.modules:
        insights = phoenix.modules['pattern_engine'].get_insights()
        if insights['most_used_commands']:
            print(f"  • Most used commands: {insights['most_used_commands'][:3]}")
        if insights['automation_opportunities']:
            print(f"  • Automation opportunities detected: {len(insights['automation_opportunities'])}")

    # Module creation demonstration
    print("\n🚀 Autonomous Module Creation Capability")
    print("=" * 60)

    if 'module_creator' in phoenix.modules:
        # Analyze need for a scheduler module
        analysis = phoenix.modules['module_creator'].analyze_need(
            "I need to schedule tasks to run at specific times",
            {'current_modules': list(phoenix.modules.keys())}
        )

        if analysis['need_detected']:
            print(f"✅ Module need detected: {analysis['module_type']}")
            print(f"   Confidence: {analysis['confidence']:.0%}")
            print(f"   Reasoning: {analysis['reasoning']}")
        else:
            print("📝 PHOENIX can create new modules when needed!")
            print("   Just ask: 'Create a module for [your need]'")

    # Save everything
    print("\n💾 Saving all learned data...")
    await phoenix.shutdown()
    print("✅ All data saved successfully!")

    print("""
    ════════════════════════════════════════════
    🎯 PHOENIX is ready for production use!

    The system now features:
    • Deep memory and context understanding
    • Self-improvement capabilities
    • Autonomous module creation
    • Pattern-based learning
    • Multi-model task routing

    Run 'python phoenix.py' for interactive mode!
    ════════════════════════════════════════════
    """)

if __name__ == "__main__":
    asyncio.run(demo_phoenix())