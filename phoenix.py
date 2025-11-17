#!/usr/bin/env python3
"""
PHOENIX AI System - Main launcher script
Run this to start your personal AI assistant.
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.phoenix_core import PhoenixCore


def check_requirements():
    """
    Check if required dependencies are installed.
    """
    missing = []

    # Check for critical dependencies
    try:
        import ollama
    except ImportError:
        missing.append("ollama")

    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")

    try:
        import psutil
    except ImportError:
        missing.append("psutil")

    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nPlease install them with:")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr install all requirements:")
        print("  pip install -r requirements.txt")
        return False

    # Check if Ollama service is running
    try:
        import ollama
        client = ollama.Client()
        client.list()
    except Exception as e:
        print("⚠️  Warning: Ollama doesn't seem to be running.")
        print("  Please start it with: ollama serve")
        print("  Or install it first: curl -fsSL https://ollama.com/install.sh | sh")
        # Don't fail, let the user continue
        return True

    return True


async def init_phoenix():
    """
    Initialize Phoenix for the first time.
    """
    print("🚀 Initializing PHOENIX for the first time...")

    # Check if Ollama has the required model
    try:
        import ollama
        client = ollama.Client()
        models = client.list()
        model_names = [m['name'] for m in models['models']]

        # Check for Qwen model
        if not any('qwen2.5:14b' in name for name in model_names):
            print("\n📦 Downloading Qwen 2.5 14B model (this may take a while)...")
            print("   This is a one-time download of about 8GB.")

            response = input("\nProceed with download? (y/n): ").strip().lower()
            if response == 'y':
                print("Pulling model...")
                # Note: This is a placeholder - actual pulling would need subprocess
                os.system("ollama pull qwen2.5:14b")
                print("✅ Model downloaded successfully!")
            else:
                print("⚠️  You can download it later with: ollama pull qwen2.5:14b")
                print("   Phoenix will use a fallback model for now.")

    except Exception as e:
        print(f"⚠️  Could not check models: {e}")

    print("\n✅ PHOENIX initialization complete!")
    print("   You can now run: python phoenix.py")


async def test_phoenix():
    """
    Run basic tests to ensure Phoenix is working.
    """
    print("🧪 Testing PHOENIX systems...")

    # Create Phoenix instance
    phoenix = PhoenixCore()

    # Test basic thinking
    print("\n1. Testing LLM connection...")
    try:
        response = await phoenix.think("Say 'Hello World' and nothing else.")
        print(f"   ✅ LLM responded: {response[:50]}...")
    except Exception as e:
        print(f"   ❌ LLM test failed: {e}")

    # Test system control
    print("\n2. Testing System Control...")
    try:
        from modules.system_control import SystemControl
        sys_control = SystemControl(phoenix.config.get('system_control', {}))

        # Test listing files
        files = sys_control.list_files(".")
        print(f"   ✅ Can list files: Found {len(files)} items")

        # Test system info
        info = sys_control.get_system_info()
        print(f"   ✅ Can get system info: CPU at {info['cpu']['percent']}%")

    except Exception as e:
        print(f"   ❌ System Control test failed: {e}")

    print("\n✅ Basic tests complete!")


async def main():
    """
    Main entry point for PHOENIX.
    """
    parser = argparse.ArgumentParser(description='PHOENIX AI System')
    parser.add_argument('--init', action='store_true', help='Initialize PHOENIX for first use')
    parser.add_argument('--test', action='store_true', help='Run system tests')
    parser.add_argument('--config', type=str, default='config/phoenix_config.yaml', help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # Check requirements first
    if not check_requirements():
        sys.exit(1)

    # Handle special commands
    if args.init:
        await init_phoenix()
        sys.exit(0)

    if args.test:
        await test_phoenix()
        sys.exit(0)

    # Normal startup
    print("""
    ╔════════════════════════════════════════════╗
    ║             PHOENIX AI SYSTEM              ║
    ║                                            ║
    ║    Personal Hybrid Operating Environment   ║
    ║      Network Intelligence eXtension        ║
    ║                                            ║
    ║            Version 0.1.0                   ║
    ╚════════════════════════════════════════════╝

    🔥 Rising from the ashes of previous attempts...
    """)

    # Create and run Phoenix
    phoenix = PhoenixCore(config_path=args.config)

    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    await phoenix.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 PHOENIX shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()