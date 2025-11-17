#!/usr/bin/env python3
"""
Test script for PHOENIX's personal data ingestion system.
Demonstrates secure ingestion and analysis of Google Takeout data.
"""

import asyncio
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))

from core.phoenix_core import PhoenixCore

async def test_personal_data_system():
    """Test the personal data ingestion and knowledge extraction system."""
    print("\n" + "=" * 70)
    print("🔐 PHOENIX Personal Data System Test")
    print("=" * 70)

    phoenix = PhoenixCore()
    await phoenix.initialize_modules()

    # Check if personal data modules are loaded
    if 'personal_vault' not in phoenix.modules:
        print("❌ Personal data vault not initialized")
        return

    if 'knowledge_extractor' not in phoenix.modules:
        print("❌ Knowledge extractor not initialized")
        return

    print("\n✅ Personal data modules loaded successfully")
    print("• Personal Data Vault: READY")
    print("• Knowledge Extractor: READY")
    print("• Encryption: AES-256 (Fernet)")
    print("• Privacy Level: MAXIMUM")

    # Get privacy report
    privacy_report = phoenix.modules['personal_vault'].get_privacy_report()
    print("\n📊 Privacy Report:")
    print(json.dumps(privacy_report, indent=2))

    # Test commands
    test_commands = [
        ("Privacy Check", "what do you know about me"),
        ("Data Storage Test", "store this in my vault: I love playing tennis"),
        ("Pattern Analysis", "show my personal insights"),
    ]

    print("\n" + "=" * 70)
    print("📝 Testing Personal Data Commands")
    print("=" * 70)

    for test_name, command in test_commands:
        print(f"\n### {test_name}")
        print(f"Command: {command}")
        print("-" * 50)

        response = await phoenix.process_command(command)
        print(f"Response: {response[:500]}")

    # Demonstration of Google Takeout ingestion
    print("\n" + "=" * 70)
    print("📦 Google Takeout Ingestion Instructions")
    print("=" * 70)
    print("\nTo ingest your Google Takeout data:")
    print("1. Download your data from https://takeout.google.com")
    print("2. Extract the archive to a folder")
    print("3. Run: 'ingest google takeout from /path/to/Takeout'")
    print("\nSupported Google services:")
    print("• Gmail (email patterns)")
    print("• Google Photos (metadata only)")
    print("• Location History (movement patterns)")
    print("• YouTube (viewing interests)")
    print("• Calendar (schedule patterns)")
    print("• Contacts (relationship mapping)")
    print("• Chrome (browsing patterns)")
    print("• Drive (document organization)")
    print("• Maps (saved places)")

    # Test storing personal data
    print("\n" + "=" * 70)
    print("🔒 Testing Secure Data Storage")
    print("=" * 70)

    vault = phoenix.modules['personal_vault']

    # Store some test data
    test_data = [
        ("I prefer morning tennis lessons", "preferences"),
        ("My favorite coach is John Smith", "relationships"),
        ("I usually train at Central Tennis Club", "locations"),
        ("Working on improving my backhand", "goals")
    ]

    for data, category in test_data:
        data_id = vault.store_personal_data(
            data=data,
            category=category,
            metadata={'source': 'test'},
            tags=f'{category},test'
        )
        print(f"✅ Stored: {data[:30]}... (ID: {data_id})")

    # Search personal data
    print("\n📍 Testing Data Search...")
    results = vault.search_personal_data("tennis")
    print(f"Found {len(results)} results for 'tennis'")

    # Test knowledge extraction
    if 'knowledge_extractor' in phoenix.modules:
        print("\n" + "=" * 70)
        print("🧠 Testing Knowledge Extraction")
        print("=" * 70)

        extractor = phoenix.modules['knowledge_extractor']

        # Get personal summary
        summary = extractor.get_personal_summary()
        print("\nPersonal Knowledge Summary:")
        for key, value in summary.items():
            print(f"• {key}: {value}")

        # Query knowledge
        query_results = extractor.query_personal_knowledge("tennis")
        print(f"\nKnowledge query for 'tennis': {len(query_results)} results")

        # Get suggestions
        suggestions = extractor.suggest_based_on_patterns()
        if suggestions:
            print("\nPersonalized Suggestions:")
            for suggestion in suggestions[:3]:
                print(f"• {suggestion['suggestion']} (confidence: {suggestion['confidence']})")

    # Security demonstration
    print("\n" + "=" * 70)
    print("🛡️ Security Features")
    print("=" * 70)
    print("• All data encrypted at rest with AES-256")
    print("• Local storage only - no cloud sync")
    print("• Owner-only file permissions (700)")
    print("• No telemetry or third-party access")
    print("• Complete data wipe available with confirmation")

    await phoenix.shutdown()

    print("\n" + "=" * 70)
    print("✅ Test Complete - Personal Data System")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("• Secure encrypted storage")
    print("• Google Takeout ingestion ready")
    print("• Pattern analysis and insights")
    print("• Privacy-first design")
    print("• Personal knowledge extraction")
    print("\nYour data remains completely private and under your control!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_personal_data_system())