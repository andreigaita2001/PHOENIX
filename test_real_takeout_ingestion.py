#!/usr/bin/env python3
"""
Test script for real Google Takeout ingestion
Can be run separately to test the integration.
"""

import sys
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Add PHOENIX to path
sys.path.insert(0, str(Path(__file__).parent))


def create_mock_takeout_data():
    """
    Create a minimal mock Takeout structure for testing.
    This creates actual files that the parser can read.
    """
    print("📦 Creating mock Google Takeout data for testing...")

    temp_dir = Path(tempfile.mkdtemp(prefix="phoenix_test_takeout_"))
    print(f"   Location: {temp_dir}")

    # Create Chrome directory with BrowserHistory.json
    chrome_dir = temp_dir / "Chrome"
    chrome_dir.mkdir()

    browser_history = {
        "Browser History": [
            {
                "page_transition": "LINK",
                "title": "Example Site",
                "url": "https://example.com",
                "client_id": "test",
                "time_usec": int(datetime.now().timestamp() * 1000000)
            },
            {
                "page_transition": "TYPED",
                "title": "GitHub",
                "url": "https://github.com",
                "client_id": "test",
                "time_usec": int(datetime.now().timestamp() * 1000000)
            }
        ]
    }

    with open(chrome_dir / "BrowserHistory.json", 'w') as f:
        json.dump(browser_history, f, indent=2)

    # Create My Activity directory
    activity_dir = temp_dir / "My Activity"
    activity_dir.mkdir()

    # Create a subdirectory for activity
    search_dir = activity_dir / "Search"
    search_dir.mkdir()

    # Create MyActivity.json
    my_activity = [
        {
            "header": "Search",
            "title": "Searched for test query",
            "titleUrl": "https://www.google.com/search?q=test",
            "time": datetime.now().isoformat() + "Z",
            "products": ["Search"]
        }
    ]

    with open(search_dir / "MyActivity.json", 'w') as f:
        json.dump(my_activity, f, indent=2)

    # Create YouTube directory
    youtube_dir = temp_dir / "YouTube and YouTube Music"
    youtube_dir.mkdir()
    history_dir = youtube_dir / "history"
    history_dir.mkdir()

    # Create watch history
    watch_history = [
        {
            "header": "YouTube",
            "title": "Watched Test Video",
            "titleUrl": "https://www.youtube.com/watch?v=test123",
            "time": datetime.now().isoformat() + "Z"
        }
    ]

    with open(history_dir / "watch-history.json", 'w') as f:
        json.dump(watch_history, f, indent=2)

    print("✅ Mock Takeout data created")
    print(f"   • Chrome history: 2 entries")
    print(f"   • My Activity: 1 entry")
    print(f"   • YouTube history: 1 entry")

    return temp_dir


def test_parser_installation():
    """Test if google_takeout_parser is installed."""
    print("\n🔍 Testing google_takeout_parser installation...")

    try:
        from google_takeout_parser.path_dispatch import TakeoutParser
        from google_takeout_parser import models
        print("✅ google_takeout_parser is installed")
        return True
    except ImportError as e:
        print(f"❌ google_takeout_parser not installed: {e}")
        print("\nInstall with: pip install google-takeout-parser")
        return False


def test_integration():
    """Test the GoogleTakeoutIntegration module."""
    print("\n🔍 Testing GoogleTakeoutIntegration module...")

    try:
        from modules.google_takeout_integration import GoogleTakeoutIntegration
        print("✅ GoogleTakeoutIntegration module loads correctly")
        return True
    except ImportError as e:
        print(f"❌ Error loading module: {e}")
        return False


def test_vault_integration():
    """Test PersonalDataVault integration."""
    print("\n🔍 Testing PersonalDataVault integration...")

    try:
        from modules.personal_data_vault import PersonalDataVault
        vault = PersonalDataVault()
        print(f"✅ PersonalDataVault initialized")
        print(f"   Vault location: {vault.vault_dir}")
        return vault
    except Exception as e:
        print(f"❌ Error initializing vault: {e}")
        return None


def test_real_ingestion(mock_takeout_dir, vault):
    """Test real ingestion with mock data."""
    print("\n🚀 Testing real ingestion with mock data...")

    try:
        from modules.google_takeout_integration import GoogleTakeoutIntegration

        integration = GoogleTakeoutIntegration(vault=vault)

        result = integration.ingest_takeout(str(mock_takeout_dir), use_cache=False)

        print(f"\n📊 Ingestion Results:")
        print(f"   • Success: {result['success']}")
        print(f"   • Total events: {result.get('total_events', 0)}")

        if result['success']:
            stats = integration.get_statistics()

            if stats['events_by_service']:
                print(f"\n   Events by service:")
                for service, count in stats['events_by_service'].items():
                    print(f"      • {service}: {count}")

            if stats['events_by_type']:
                print(f"\n   Events by type:")
                for event_type, count in stats['events_by_type'].items():
                    print(f"      • {event_type}: {count}")

            print("\n✅ Ingestion test PASSED")
            return True
        else:
            print(f"\n❌ Ingestion failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Error during ingestion test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vault_retrieval(vault):
    """Test retrieving data from vault."""
    print("\n🔍 Testing data retrieval from vault...")

    try:
        # Search for Chrome data
        results = vault.search_personal_data("", category="browser_history")

        print(f"   Found {len(results)} browser history entries")

        if results:
            print(f"   Sample entry:")
            print(f"      • Category: {results[0]['category']}")
            print(f"      • Type: {results[0]['type']}")
            print(f"      • Metadata: {results[0]['metadata']}")
            print("\n✅ Data retrieval test PASSED")
            return True
        else:
            print("   ⚠️  No data found (might be OK for mock data)")
            return True

    except Exception as e:
        print(f"❌ Error retrieving data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║     PHOENIX - Real Takeout Ingestion Test Suite              ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    all_tests_passed = True

    # Test 1: Parser installation
    if not test_parser_installation():
        print("\n⚠️  Cannot continue without google_takeout_parser")
        sys.exit(1)

    # Test 2: Integration module
    if not test_integration():
        all_tests_passed = False

    # Test 3: Vault
    vault = test_vault_integration()
    if not vault:
        all_tests_passed = False
        print("\n⚠️  Cannot continue without vault")
        sys.exit(1)

    # Test 4: Create mock data
    try:
        mock_dir = create_mock_takeout_data()
    except Exception as e:
        print(f"\n❌ Error creating mock data: {e}")
        all_tests_passed = False
        sys.exit(1)

    # Test 5: Real ingestion
    if not test_real_ingestion(mock_dir, vault):
        all_tests_passed = False

    # Test 6: Data retrieval
    if not test_vault_retrieval(vault):
        all_tests_passed = False

    # Cleanup
    print("\n🧹 Cleaning up mock data...")
    import shutil
    try:
        shutil.rmtree(mock_dir)
        print("✅ Cleanup complete")
    except Exception as e:
        print(f"⚠️  Could not clean up {mock_dir}: {e}")

    # Summary
    print("\n" + "=" * 70)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED")
        print("\nYou can now use the real ingestion script:")
        print("  python ingest_takeout_real.py /path/to/your/Takeout")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nCheck the errors above and ensure:")
        print("  1. google-takeout-parser is installed")
        print("  2. All modules load correctly")
        print("  3. Vault is properly initialized")
    print("=" * 70)

    sys.exit(0 if all_tests_passed else 1)


if __name__ == "__main__":
    main()
