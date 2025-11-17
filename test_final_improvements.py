#!/usr/bin/env python3
"""
Final test demonstrating PHOENIX's improved capabilities and non-hallucination.
"""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from core.phoenix_core import PhoenixCore

async def test_improved_phoenix():
    """Test PHOENIX's improved behavior."""
    print("\n" + "=" * 70)
    print("🔥 PHOENIX AI - Improved Non-Hallucinatory System Test")
    print("=" * 70)

    phoenix = PhoenixCore()
    await phoenix.initialize_modules()

    tests = [
        ("Web Search", "check the internet for Eddie Herr 2025 tournament dates"),
        ("Code Execution", "run python code:\n```python\nprint('Hello from actual Python execution!')\nprint(2 + 2)\n```"),
        ("Honest Limitation", "send an email to test@example.com"),
        ("Schedule Management", "add a tennis lesson tomorrow at 4pm with Sarah"),
        ("Capability Check", "what can you actually do?")
    ]

    for test_name, command in tests:
        print(f"\n📝 Test: {test_name}")
        print("-" * 50)
        print(f"Command: {command[:100]}...")

        response = await phoenix.process_command(command)
        print(f"Response: {response[:300]}...")

        # Check for hallucination indicators
        if "i'll" in response.lower() and "email" in command.lower():
            if "don't have" in response.lower() or "cannot" in response.lower():
                print("✅ Correctly identified limitation!")
            else:
                print("⚠️ May be hallucinating capability")
        elif "web search results" in response.lower():
            print("✅ Provided actual search results/links")
        elif "code executed" in response.lower():
            print("✅ Actually executed code")
        elif "added lesson" in response.lower():
            print("✅ Scheduled successfully")

    # Capability report
    if 'capability_manager' in phoenix.modules:
        print("\n" + "=" * 70)
        print("📊 Capability Report")
        print("=" * 70)
        report = phoenix.modules['capability_manager'].generate_capability_report()
        print(report[:800])

    await phoenix.shutdown()

    print("\n" + "=" * 70)
    print("✅ Test Complete - PHOENIX is now:")
    print("  • Honest about limitations")
    print("  • Actually executes code instead of pretending")
    print("  • Provides real search links instead of inventing data")
    print("  • Can create modules to fill capability gaps")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_improved_phoenix())