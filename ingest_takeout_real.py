#!/usr/bin/env python3
"""
REAL Google Takeout Ingestion Script for PHOENIX
Uses the actual google_takeout_parser library for real data ingestion.

This replaces the placeholder/fluff code with working implementation.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import argparse

# Add PHOENIX to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.personal_data_vault import PersonalDataVault
from modules.google_takeout_integration import GoogleTakeoutIntegration


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    log_dir = Path.home() / '.phoenix_vault' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f'takeout_ingestion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger("PHOENIX.TakeoutIngestion")


def print_banner():
    """Print welcome banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║        PHOENIX - Real Google Takeout Ingestion                ║
║                                                               ║
║  Using google_takeout_parser library for actual parsing       ║
║  All data encrypted and stored locally                        ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def verify_takeout_path(takeout_path: Path) -> bool:
    """
    Verify the Takeout path exists and looks valid.

    Args:
        takeout_path: Path to Takeout directory

    Returns:
        True if valid, False otherwise
    """
    if not takeout_path.exists():
        print(f"❌ Error: Path does not exist: {takeout_path}")
        return False

    if not takeout_path.is_dir():
        print(f"❌ Error: Path is not a directory: {takeout_path}")
        return False

    # Check for common Google Takeout directories
    expected_dirs = [
        'Chrome', 'Gmail', 'Google Photos', 'Location History',
        'YouTube and YouTube Music', 'My Activity', 'Calendar',
        'Contacts', 'Drive', 'Maps'
    ]

    found_dirs = [d for d in expected_dirs if (takeout_path / d).exists()]

    if not found_dirs:
        print(f"⚠️  Warning: No common Google Takeout directories found in {takeout_path}")
        print(f"   Expected one of: {', '.join(expected_dirs)}")
        print(f"   Found subdirectories: {[d.name for d in takeout_path.iterdir() if d.is_dir()]}")

        response = input("\n   Continue anyway? (y/n): ")
        return response.lower() == 'y'

    print(f"✅ Found {len(found_dirs)} Google services:")
    for service in found_dirs:
        print(f"   • {service}")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Ingest Google Takeout data into PHOENIX using real parser'
    )
    parser.add_argument(
        'takeout_path',
        type=str,
        help='Path to extracted Google Takeout folder (containing subdirs like Gmail, Chrome, etc.)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable parser cache (slower but forces fresh parse)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logging(args.verbose)
    print_banner()

    # Verify path
    takeout_path = Path(args.takeout_path).expanduser().resolve()
    print(f"\n📁 Takeout Path: {takeout_path}\n")

    if not verify_takeout_path(takeout_path):
        sys.exit(1)

    # Initialize vault
    print("\n🔐 Initializing encrypted vault...")
    try:
        vault = PersonalDataVault()
        print(f"✅ Vault ready at: {vault.vault_dir}")
    except Exception as e:
        print(f"❌ Error initializing vault: {e}")
        logger.error(f"Vault initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # Initialize integration
    print("\n🔧 Initializing Google Takeout parser...")
    try:
        integration = GoogleTakeoutIntegration(vault=vault)
        print("✅ Parser ready")
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease install google_takeout_parser:")
        print("  pip install google-takeout-parser")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error initializing integration: {e}")
        logger.error(f"Integration initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # Confirm
    print("\n" + "=" * 70)
    print("Ready to ingest your Google Takeout data")
    print("=" * 70)
    print("\nThis will:")
    print("  ✅ Parse all data from Takeout using google_takeout_parser")
    print("  ✅ Encrypt and store locally in your vault")
    print("  ✅ Never send data to the cloud")
    print("  ✅ Use caching for faster subsequent runs" if not args.no_cache else "  ⚠️  Cache disabled - will be slower")

    print(f"\nNote: First run may take several minutes to parse all data.")

    response = input("\n🤔 Proceed with ingestion? (yes/no): ").strip().lower()
    if response != 'yes':
        print("\n❌ Ingestion cancelled")
        sys.exit(0)

    # Start ingestion
    print("\n" + "=" * 70)
    print("🚀 Starting ingestion...")
    print("=" * 70 + "\n")

    start_time = datetime.now()

    try:
        result = integration.ingest_takeout(
            str(takeout_path),
            use_cache=not args.no_cache
        )

        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 70)
        if result['success']:
            print("✨ INGESTION COMPLETE!")
        else:
            print("⚠️  INGESTION COMPLETED WITH ERRORS")
        print("=" * 70)

        # Print statistics
        print(f"\n📊 Results:")
        print(f"   • Total events: {result.get('total_events', 0):,}")
        print(f"   • Time taken: {duration}")
        print(f"   • Vault location: {vault.vault_dir}")

        # Get detailed stats
        stats = integration.get_statistics()

        if stats['events_by_service']:
            print(f"\n📈 Events by service:")
            for service, count in sorted(stats['events_by_service'].items(), key=lambda x: x[1], reverse=True):
                print(f"   • {service:<20} {count:>6,} events")

        if stats['events_by_type']:
            print(f"\n🔍 Events by type:")
            for event_type, count in sorted(stats['events_by_type'].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   • {event_type:<30} {count:>6,} events")

        if stats['errors'] > 0:
            print(f"\n⚠️  Errors encountered: {stats['errors']}")
            if args.verbose and stats['error_details']:
                print("   First few errors:")
                for err in stats['error_details'][:5]:
                    print(f"   • {err}")

        # Privacy report
        print(f"\n🔐 Privacy Report:")
        privacy = vault.get_privacy_report()
        print(f"   • Encryption: {privacy['encryption']}")
        print(f"   • Storage: {privacy['storage_location']}")
        print(f"   • Access: Owner only")
        print(f"   • Data stays local: ✅")
        print(f"   • Cloud access: ❌")

        print(f"\n🎉 Your PHOENIX now has access to your personal data!")
        print(f"   Run 'python phoenix.py' to interact with your personalized AI")

        # Log file location
        print(f"\n📝 Full log saved to:")
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                print(f"   {handler.baseFilename}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Ingestion interrupted by user")
        print("   Progress has been saved and can be resumed")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
