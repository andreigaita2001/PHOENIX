#!/usr/bin/env python3
"""
Test script to verify PHOENIX doesn't hallucinate and actually executes actions.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.phoenix_core import PhoenixCore

async def test_no_hallucination():
    """Test that PHOENIX doesn't hallucinate capabilities."""
    print("\n🔥 Testing PHOENIX Non-Hallucination System")
    print("=" * 60)

    # Initialize PHOENIX
    phoenix = PhoenixCore()
    await phoenix.initialize_modules()

    # Test 1: Web Search (should work now)
    print("\n📝 Test 1: Web Search Capability")
    print("-" * 40)
    response = await phoenix.process_command("search the web for Python tutorials")
    print(f"Response: {response[:300]}...")
    if "don't have web search capability" in response:
        print("❌ Web search not working")
    elif "Web Search Results" in response:
        print("✅ Web search executed successfully!")
    else:
        print("⚠️ Unexpected response")

    # Test 2: GUI Creation (should work now)
    print("\n📝 Test 2: GUI Creation")
    print("-" * 40)
    response = await phoenix.process_command("show my schedule in a graphical interface")
    print(f"Response: {response[:300]}...")
    if "cannot create GUI" in response:
        print("❌ GUI creation failed")
    elif "window opened" in response.lower() or "gui" in response.lower():
        print("✅ GUI creation executed!")
    else:
        print("⚠️ Unexpected response")

    # Test 3: Scheduling (should work)
    print("\n📝 Test 3: Tennis Scheduling")
    print("-" * 40)
    response = await phoenix.process_command("add a tennis lesson tomorrow at 3pm with John Smith")
    print(f"Response: {response[:300]}...")
    if "don't have" in response.lower() or "cannot" in response.lower():
        print("❌ Scheduling failed")
    elif "added lesson" in response.lower() or "scheduled" in response.lower():
        print("✅ Scheduling works!")
    else:
        print("⚠️ Unexpected response")

    # Test 4: Capability Awareness
    print("\n📝 Test 4: Capability Awareness")
    print("-" * 40)
    if 'capability_manager' in phoenix.modules:
        report = phoenix.modules['capability_manager'].generate_capability_report()
        print("Capability Report Generated:")
        print(report[:500] + "...")
        print("✅ Capability awareness active!")
    else:
        print("❌ Capability manager not loaded")

    # Test 5: Check for hallucination
    print("\n📝 Test 5: Hallucination Check")
    print("-" * 40)
    response = await phoenix.process_command("send an email to my client")
    print(f"Response: {response[:300]}...")
    if "cannot" in response.lower() or "don't have" in response.lower() or "need" in response.lower():
        print("✅ Correctly identified missing capability!")
    else:
        print("❌ May be hallucinating email capability")

    # Shutdown
    await phoenix.shutdown()
    print("\n" + "=" * 60)
    print("🔥 Testing Complete!")

if __name__ == "__main__":
    asyncio.run(test_no_hallucination())