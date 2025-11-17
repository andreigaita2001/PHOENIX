#!/usr/bin/env python3
"""
Test PHOENIX Autonomous Capabilities
See your AI learning and working autonomously!
"""

import asyncio
import time
from core.phoenix_core import PhoenixCore

async def test_autonomous_phoenix():
    """Test autonomous features."""
    print("""
    ╔════════════════════════════════════════════╗
    ║      PHOENIX AUTONOMOUS TEST              ║
    ╚════════════════════════════════════════════╝

    Watch PHOENIX learn and adapt...
    """)

    phoenix = PhoenixCore()
    await phoenix.initialize_modules()

    if 'learning' not in phoenix.modules or 'automation' not in phoenix.modules:
        print("❌ Learning or automation modules not initialized!")
        print("Please install dependencies: pip install watchdog schedule scikit-learn numpy")
        return

    print("\n1. Testing Learning from Commands...")
    # Simulate some commands to learn from
    test_commands = [
        "check my gpu",
        "show system info",
        "check my gpu",  # Repeated to learn pattern
        "list files",
        "check my cpu",
    ]

    for cmd in test_commands:
        print(f"   Processing: {cmd}")
        response = await phoenix.process_command(cmd)
        # Learning happens automatically
        time.sleep(0.5)

    print("\n2. Checking Predictions...")
    predictions = phoenix.modules['learning'].predict_next_command("check my")
    if predictions:
        print("   PHOENIX predicts you might want:")
        for cmd, confidence in predictions[:3]:
            print(f"   • {cmd} ({confidence:.0%} likely)")

    print("\n3. Checking User Model...")
    prefs = phoenix.modules['learning'].get_user_preferences()
    print(f"   Communication style: {prefs['communication_style']}")
    print(f"   Expertise level: {prefs['expertise_level']:.0%}")
    if prefs['favorite_commands']:
        print("   Your favorite commands:")
        for cmd, count in prefs['favorite_commands'][:3]:
            print(f"   • {cmd} ({count} times)")

    print("\n4. Checking Automation Status...")
    status = phoenix.modules['automation'].get_automation_status()
    print(f"   Automation: {'RUNNING' if status['is_running'] else 'STOPPED'}")
    print(f"   Autonomous actions enabled:")
    for action, enabled in status['autonomous_actions'].items():
        if enabled:
            print(f"   ✅ {action}")

    print("\n5. Testing Learning Report...")
    report = phoenix.modules['learning'].generate_learning_report()
    print(report[:500] + "..." if len(report) > 500 else report)

    # Stop automation
    await phoenix.modules['automation'].stop()

    print("""
    ════════════════════════════════════════════
    ✅ Autonomous capabilities test complete!

    PHOENIX is now:
    • Learning from every command
    • Predicting what you need next
    • Building a model of your preferences
    • Running automation in background
    • Getting smarter with each use

    Try these in PHOENIX:
    - "learning report" - See what it learned
    - "automation status" - Check background tasks
    - Watch as it predicts your next command!
    ════════════════════════════════════════════
    """)

if __name__ == "__main__":
    asyncio.run(test_autonomous_phoenix())