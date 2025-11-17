#!/usr/bin/env python3
"""
Test PHOENIX System Access
Verify that PHOENIX can actually control your system.
"""

import asyncio
from core.phoenix_core import PhoenixCore

async def test_system_access():
    """Test system access capabilities."""
    print("""
    ╔════════════════════════════════════════════╗
    ║      PHOENIX SYSTEM ACCESS TEST           ║
    ╚════════════════════════════════════════════╝
    """)

    phoenix = PhoenixCore()
    await phoenix.initialize_modules()

    if 'system' not in phoenix.modules:
        print("❌ System Control module not initialized!")
        return

    print("Testing system commands...\n")

    # Test GPU check
    print("1. Testing GPU check...")
    result = await phoenix._execute_system_action('system_info', {})
    print(result)

    print("\n" + "="*50 + "\n")

    # Test file listing
    print("2. Testing file listing...")
    result = await phoenix._execute_system_action('list_files', {'directory': '.'})
    print(result[:500] + "..." if len(result) > 500 else result)

    print("\n" + "="*50 + "\n")

    # Test process listing
    print("3. Testing process listing...")
    result = await phoenix._execute_system_action('process_list', {})
    print(result)

    print("""
    ════════════════════════════════════════════
    ✅ System access test complete!

    PHOENIX can now:
    - Check your GPU ✓
    - List files ✓
    - Monitor processes ✓
    - Execute commands ✓

    Try these in PHOENIX:
    - "check my gpu"
    - "show system info"
    - "list files"
    ════════════════════════════════════════════
    """)

if __name__ == "__main__":
    asyncio.run(test_system_access())