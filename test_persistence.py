#!/usr/bin/env python3
"""
Test script for PHOENIX conversation persistence and module integration.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from core.phoenix_core import PhoenixCore

async def test_phoenix():
    """Test PHOENIX with conversation persistence."""
    print("\n🔥 Testing PHOENIX AI System - Conversation Persistence")
    print("=" * 60)

    # Initialize PHOENIX
    phoenix = PhoenixCore()

    # Initialize modules directly (since we won't run the main loop)
    await phoenix.initialize_modules()

    print("\n📝 Test 1: Basic conversation memory")
    print("-" * 40)

    # Test 1: Initial greeting
    response1 = await phoenix.process_command("Hello, my name is TestUser")
    print(f"User: Hello, my name is TestUser")
    print(f"PHOENIX: {response1[:200]}...")

    # Test 2: Check if it remembers the name
    print("\n📝 Test 2: Memory recall")
    print("-" * 40)
    response2 = await phoenix.process_command("What is my name?")
    print(f"User: What is my name?")
    print(f"PHOENIX: {response2[:200]}...")

    # Check if name was remembered
    if "TestUser" in response2 or "test" in response2.lower():
        print("✅ PHOENIX remembered the user's name!")
    else:
        print("❌ PHOENIX did not remember the user's name")

    # Test 3: Pattern learning
    print("\n📝 Test 3: Pattern recognition")
    print("-" * 40)

    # Simulate repeated command pattern
    for i in range(3):
        await phoenix.process_command(f"check system status")
        print(f"Recorded command #{i+1}: check system status")

    # Check if pattern was learned
    if 'pattern_engine' in phoenix.modules:
        insights = phoenix.modules['pattern_engine'].get_insights()
        if insights['most_used_commands']:
            print(f"✅ Pattern detected: Most used commands: {insights['most_used_commands'][:3]}")
        else:
            print("❌ No patterns detected yet")

    # Test 4: Module integration check
    print("\n📝 Test 4: Module integration")
    print("-" * 40)

    modules = list(phoenix.modules.keys())
    print(f"Loaded modules: {modules}")

    critical_modules = ['memory', 'safety', 'multi_model', 'pattern_engine', 'module_creator']
    for module in critical_modules:
        if module in modules:
            print(f"✅ {module} initialized")
        else:
            print(f"❌ {module} not initialized")

    # Test 5: Memory persistence
    print("\n📝 Test 5: Memory statistics")
    print("-" * 40)

    if 'memory' in phoenix.modules:
        stats = phoenix.modules['memory'].get_stats()
        print(f"Memory stats: {stats}")
        if stats['total_conversations'] > 0:
            print(f"✅ Conversations stored: {stats['total_conversations']}")
        else:
            print("❌ No conversations stored")

    # Shutdown and save
    print("\n📝 Shutting down and saving state...")
    print("-" * 40)
    await phoenix.shutdown()
    print("✅ PHOENIX shutdown complete")

    print("\n" + "=" * 60)
    print("🔥 PHOENIX Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_phoenix())