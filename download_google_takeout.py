#!/usr/bin/env python3
"""
Google Takeout Download Helper
Helps download and prepare Google Takeout data for PHOENIX ingestion.
"""

import os
import sys
from pathlib import Path
import subprocess
import hashlib
from datetime import datetime

class GoogleTakeoutDownloader:
    """
    Helper to download and prepare Google Takeout data.
    """

    def __init__(self):
        self.download_dir = Path.home() / 'Downloads' / 'GoogleTakeout'
        self.extract_dir = Path.home() / 'GoogleTakeout_Extracted'

    def print_banner(self):
        """Print welcome banner."""
        print("""
╔════════════════════════════════════════════════════════════╗
║     PHOENIX - Google Takeout Download Helper              ║
║                                                            ║
║  This tool helps you download and prepare your 148GB      ║
║  of Google Takeout data for local AI ingestion.           ║
╚════════════════════════════════════════════════════════════╝
        """)

    def show_download_instructions(self):
        """Show instructions for downloading from email."""
        print("\n📧 STEP 1: Download from Email")
        print("=" * 60)
        print("""
Your Google Takeout data is in your email. Here's how to download it:

OPTION A: Direct Download (Recommended)
  1. Open the Google Takeout email
  2. Click on each download link
  3. Save all files to: {}
  4. Come back here when download is complete

OPTION B: Using Gmail API (Advanced)
  1. We can help you set up automatic download
  2. Requires Gmail API credentials
  3. More complex but hands-off

Which option would you like?
  [A] Manual download (I'll do it myself)
  [B] Set up automatic download
  [Q] Quit
""".format(self.download_dir))

    def create_download_directory(self):
        """Create directory for downloads."""
        self.download_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✅ Download directory ready: {self.download_dir}")

    def wait_for_downloads(self):
        """Wait for user to complete downloads."""
        print("\n⏳ Waiting for downloads...")
        print(f"\nPlease download all Google Takeout files to:")
        print(f"  {self.download_dir}")
        print("\nPress ENTER when all downloads are complete...")
        input()

    def verify_downloads(self):
        """Verify downloaded files."""
        print("\n🔍 Checking downloaded files...")

        files = list(self.download_dir.glob("takeout-*.zip")) + \
                list(self.download_dir.glob("takeout-*.tgz"))

        if not files:
            print("❌ No Google Takeout files found!")
            print(f"   Looking in: {self.download_dir}")
            print("\nExpected files like:")
            print("  • takeout-20241117-001.zip")
            print("  • takeout-20241117-002.zip")
            print("  • etc...")
            return False

        print(f"\n✅ Found {len(files)} file(s)")

        total_size = 0
        for f in files:
            size_gb = f.stat().st_size / (1024**3)
            total_size += size_gb
            print(f"  • {f.name} ({size_gb:.2f} GB)")

        print(f"\n📊 Total size: {total_size:.2f} GB")

        if total_size < 100:
            print("\n⚠️  Warning: Total size is less than expected (148GB)")
            print("   Are all files downloaded?")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False

        return True

    def extract_archives(self):
        """Extract all Takeout archives."""
        print("\n📦 STEP 2: Extracting Archives")
        print("=" * 60)

        self.extract_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(list(self.download_dir.glob("takeout-*.zip")) + \
                      list(self.download_dir.glob("takeout-*.tgz")))

        print(f"Extracting {len(files)} archive(s) to:")
        print(f"  {self.extract_dir}\n")

        for i, archive_file in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Extracting {archive_file.name}...")

            try:
                if archive_file.suffix == '.zip':
                    # Use unzip command
                    subprocess.run(
                        ['unzip', '-q', '-o', str(archive_file), '-d', str(self.extract_dir)],
                        check=True
                    )
                elif archive_file.suffix == '.tgz' or archive_file.name.endswith('.tar.gz'):
                    # Use tar command
                    subprocess.run(
                        ['tar', '-xzf', str(archive_file), '-C', str(self.extract_dir)],
                        check=True
                    )

                print(f"  ✅ Complete")

            except subprocess.CalledProcessError as e:
                print(f"  ❌ Error: {e}")
                print("  Trying alternative extraction method...")

                # Fallback to Python's zipfile/tarfile
                if archive_file.suffix == '.zip':
                    import zipfile
                    with zipfile.ZipFile(archive_file, 'r') as zip_ref:
                        zip_ref.extractall(self.extract_dir)
                    print(f"  ✅ Complete (using Python zipfile)")
                else:
                    import tarfile
                    with tarfile.open(archive_file, 'r:gz') as tar_ref:
                        tar_ref.extractall(self.extract_dir)
                    print(f"  ✅ Complete (using Python tarfile)")

        print("\n✅ All archives extracted!")

    def verify_extraction(self):
        """Verify extraction was successful."""
        print("\n🔍 Verifying extracted data...")

        # Look for Takeout folder
        takeout_folders = list(self.extract_dir.glob("**/Takeout"))

        if not takeout_folders:
            # Sometimes it extracts directly
            takeout_folders = [self.extract_dir]

        takeout_path = takeout_folders[0]
        print(f"\n📁 Takeout data location: {takeout_path}")

        # Check for expected Google service folders
        expected_services = [
            'Gmail', 'Google Photos', 'Location History',
            'YouTube and YouTube Music', 'Calendar', 'Chrome',
            'Drive', 'Maps'
        ]

        found_services = []
        for service in expected_services:
            service_path = takeout_path / service
            if service_path.exists():
                found_services.append(service)

                # Get size
                size = sum(f.stat().st_size for f in service_path.rglob('*') if f.is_file())
                size_gb = size / (1024**3)
                print(f"  ✅ {service} ({size_gb:.2f} GB)")

        if not found_services:
            print("\n❌ No Google services found in extraction")
            return None

        print(f"\n✅ Found {len(found_services)} Google services")

        return takeout_path

    def prepare_for_ingestion(self, takeout_path):
        """Prepare final ingestion command."""
        print("\n🚀 STEP 3: Ready for Ingestion!")
        print("=" * 60)

        print(f"""
Your Google Takeout data is ready for PHOENIX ingestion!

Data location: {takeout_path}

To ingest into PHOENIX, run:

  python ingest_takeout.py "{takeout_path}"

Or from within PHOENIX:

  python phoenix.py
  > ingest my google takeout from "{takeout_path}"

This will:
  • Securely encrypt and store all your data locally
  • Analyze patterns in your emails, photos, location, etc.
  • Extract personal insights and preferences
  • Build a knowledge graph about you
  • Keep everything private and local (no cloud!)

Estimated ingestion time: 2-6 hours (depending on your CPU)

Ready to proceed? The ingestion script will be created next.
""")

        return takeout_path

    def run(self):
        """Run the download helper."""
        self.print_banner()

        # Step 1: Download instructions
        self.show_download_instructions()

        choice = input("Choose [A/B/Q]: ").strip().upper()

        if choice == 'Q':
            print("\n👋 Goodbye!")
            return

        elif choice == 'A':
            # Manual download
            self.create_download_directory()
            self.wait_for_downloads()

            if not self.verify_downloads():
                print("\n❌ Download verification failed")
                return

            # Extract
            self.extract_archives()

            # Verify
            takeout_path = self.verify_extraction()
            if not takeout_path:
                print("\n❌ Extraction verification failed")
                return

            # Prepare
            self.prepare_for_ingestion(takeout_path)

        elif choice == 'B':
            # Automatic download
            print("\n🔧 Gmail API setup coming soon...")
            print("For now, please use Option A (manual download)")
            return

        else:
            print("\n❌ Invalid choice")
            return

        print("\n✅ Setup complete!")
        print("    Next step: Run the ingestion script")


def main():
    """Main entry point."""
    downloader = GoogleTakeoutDownloader()
    downloader.run()


if __name__ == "__main__":
    main()
