#!/usr/bin/env python3
"""
Test PHOENIX Memory Capabilities
Run this to see the memory system in action!
"""

import asyncio
from core.phoenix_core import PhoenixCore

async def test_memory():
    """Test memory features interactively."""
    print("""
    ╔════════════════════════════════════════════╗
    ║         PHOENIX MEMORY TEST                ║
    ╚════════════════════════════════════════════╝

    Testing memory capabilities...
    """)

    phoenix = PhoenixCore()
    await phoenix.initialize_modules()

    if 'memory' not in phoenix.modules:
        print("❌ Memory module not initialized!")
        return

    memory = phoenix.modules['memory']

    # Test conversation memory
    print("1. Testing conversation memory...")
    memory.remember_conversation("Hello Phoenix!", "Hello! I'm ready to assist you.")
    memory.remember_conversation("What's 2+2?", "2+2 equals 4.")
    print("   ✅ Stored 2 conversations")

    # Test learning facts
    print("\n2. Testing fact learning...")
    memory.learn_fact("User prefers Python programming", "preferences")
    memory.learn_fact("User is learning AI development", "interests")
    print("   ✅ Learned 2 facts")

    # Test user profile
    print("\n3. Testing user profile...")
    memory.update_user_profile("name", "AI Developer")
    memory.update_user_profile("goal", "Build self-improving AI")
    profile = memory.get_user_info()
    print(f"   ✅ Profile: {profile.get('name')} - Goal: {profile.get('goal')}")

    # Test similarity search
    print("\n4. Testing memory recall...")
    similar = memory.recall_similar("math calculation", limit=2)
    print(f"   ✅ Found {len(similar)} similar memories")

    # Show stats
    print("\n5. Memory Statistics:")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"   - {key}: {value}")

    # Save memory
    print("\n6. Saving memory to disk...")
    memory.save()
    print("   ✅ Memory saved to data/")

    print("""
    ════════════════════════════════════════════
    Memory test complete!

    PHOENIX now remembers:
    - All conversations
    - Facts you teach it
    - Your preferences
    - Past interactions

    Try these commands in PHOENIX:
    - "my name is [your name]"
    - "remember that [fact]"
    - "what do you remember?"
    ════════════════════════════════════════════
    """)

if __name__ == "__main__":
    asyncio.run(test_memory())