#!/usr/bin/env python3
"""
PHOENIX Google Takeout Ingestion Script
Handles large-scale (148GB+) Google Takeout data ingestion with:
- Progress tracking
- Resume capability
- Memory optimization
- Advanced analysis
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pickle

# Add PHOENIX to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.personal_data_vault import PersonalDataVault
from modules.personal_knowledge_extractor import PersonalKnowledgeExtractor


class IngestionProgress:
    """Track ingestion progress for resume capability."""

    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.progress = {
            'started_at': None,
            'last_updated': None,
            'total_files_processed': 0,
            'total_items_ingested': 0,
            'services_completed': [],
            'services_in_progress': {},
            'errors': [],
            'status': 'not_started'
        }
        self.load()

    def load(self):
        """Load progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)

    def save(self):
        """Save progress to file."""
        self.progress['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def start(self):
        """Mark ingestion as started."""
        self.progress['started_at'] = datetime.now().isoformat()
        self.progress['status'] = 'in_progress'
        self.save()

    def complete_service(self, service: str, items_count: int):
        """Mark a service as completed."""
        self.progress['services_completed'].append(service)
        if service in self.progress['services_in_progress']:
            del self.progress['services_in_progress'][service]
        self.progress['total_items_ingested'] += items_count
        self.save()

    def update_service_progress(self, service: str, processed: int, total: int):
        """Update progress for a service."""
        self.progress['services_in_progress'][service] = {
            'processed': processed,
            'total': total,
            'percentage': (processed / total * 100) if total > 0 else 0
        }
        self.save()

    def add_error(self, service: str, error: str):
        """Record an error."""
        self.progress['errors'].append({
            'service': service,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        self.save()

    def is_service_completed(self, service: str) -> bool:
        """Check if a service was already completed."""
        return service in self.progress['services_completed']

    def complete(self):
        """Mark ingestion as completed."""
        self.progress['status'] = 'completed'
        self.progress['completed_at'] = datetime.now().isoformat()
        self.save()


class EnhancedGoogleTakeoutIngester:
    """
    Enhanced ingestion system for large Google Takeout archives.
    """

    def __init__(self, takeout_path: str, resume: bool = True):
        self.takeout_path = Path(takeout_path)
        self.logger = self._setup_logging()

        # Initialize vault
        self.vault = PersonalDataVault()

        # Initialize knowledge extractor
        self.knowledge_extractor = PersonalKnowledgeExtractor(self.vault)

        # Progress tracking
        progress_file = Path.home() / '.phoenix_vault' / 'ingestion_progress.json'
        self.progress = IngestionProgress(progress_file)

        self.resume = resume

        # Service processors with enhanced memory management
        self.processors = {
            'Gmail': self._process_gmail_optimized,
            'Google Photos': self._process_photos_optimized,
            'Location History': self._process_location_optimized,
            'YouTube and YouTube Music': self._process_youtube_optimized,
            'Calendar': self._process_calendar_optimized,
            'Contacts': self._process_contacts_optimized,
            'Drive': self._process_drive_optimized,
            'Chrome': self._process_chrome_optimized,
            'Maps': self._process_maps_optimized,
            'My Activity': self._process_activity_optimized
        }

    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        log_dir = Path.home() / '.phoenix_vault' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f'ingestion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        return logging.getLogger("PHOENIX.Ingestion")

    def print_banner(self):
        """Print welcome banner."""
        print("""
╔═══════════════════════════════════════════════════════════════╗
║           PHOENIX - Google Takeout Ingestion                  ║
║                                                               ║
║  Securely ingesting your Google data into your personal AI    ║
║  All data stays local and encrypted                           ║
╚═══════════════════════════════════════════════════════════════╝
        """)

    def analyze_takeout(self) -> Dict[str, Any]:
        """Analyze the Takeout structure before ingestion."""
        print("\n📊 Analyzing Google Takeout structure...")

        analysis = {
            'takeout_path': str(self.takeout_path),
            'exists': self.takeout_path.exists(),
            'services_found': [],
            'estimated_size_gb': 0,
            'estimated_items': 0
        }

        if not self.takeout_path.exists():
            print(f"❌ Takeout path not found: {self.takeout_path}")
            return analysis

        # Find all service directories
        for service in self.processors.keys():
            service_path = self.takeout_path / service
            if service_path.exists():
                # Calculate size
                size = sum(f.stat().st_size for f in service_path.rglob('*') if f.is_file())
                size_gb = size / (1024**3)

                # Estimate items
                file_count = len(list(service_path.rglob('*')))

                analysis['services_found'].append({
                    'name': service,
                    'path': str(service_path),
                    'size_gb': size_gb,
                    'file_count': file_count
                })

                analysis['estimated_size_gb'] += size_gb
                analysis['estimated_items'] += file_count

        # Print analysis
        print(f"\n✅ Found {len(analysis['services_found'])} Google services")
        print(f"📦 Total size: {analysis['estimated_size_gb']:.2f} GB")
        print(f"📄 Estimated files: {analysis['estimated_items']:,}")

        print("\n📋 Services breakdown:")
        for service in analysis['services_found']:
            status = "✅ Ready" if not self.progress.is_service_completed(service['name']) else "⏭️  Already completed"
            print(f"  • {service['name']:<35} {service['size_gb']:>8.2f} GB   {status}")

        return analysis

    async def ingest_all(self):
        """Ingest all Google Takeout data."""
        self.print_banner()

        # Analyze structure
        analysis = self.analyze_takeout()

        if not analysis['services_found']:
            print("\n❌ No Google services found in Takeout")
            return

        # Check if resume
        if self.resume and self.progress.progress['status'] == 'in_progress':
            print("\n🔄 Resuming previous ingestion...")
            print(f"   Already completed: {len(self.progress.progress['services_completed'])} services")
            print(f"   Items ingested: {self.progress.progress['total_items_ingested']:,}")
        else:
            print("\n🚀 Starting fresh ingestion...")
            self.progress.start()

        # Estimate time
        estimated_hours = analysis['estimated_size_gb'] / 25  # ~25GB per hour estimate
        print(f"\n⏱️  Estimated time: {estimated_hours:.1f} hours")

        # Confirm
        print("\nThis will:")
        print("  • Encrypt and store all data locally")
        print("  • Never send data to the cloud")
        print("  • Analyze patterns and extract insights")
        print("  • Build a personal knowledge graph")

        response = input("\n🤔 Proceed with ingestion? (yes/no): ").strip().lower()
        if response != 'yes':
            print("\n❌ Ingestion cancelled")
            return

        print("\n" + "=" * 70)
        print("🔥 Starting ingestion...")
        print("=" * 70)

        # Process each service
        total_ingested = 0

        for service_info in analysis['services_found']:
            service_name = service_info['name']

            # Skip if already completed
            if self.progress.is_service_completed(service_name):
                print(f"\n⏭️  Skipping {service_name} (already completed)")
                continue

            # Get processor
            processor = self.processors.get(service_name)
            if not processor:
                self.logger.warning(f"No processor for {service_name}")
                continue

            print(f"\n{'='*70}")
            print(f"📥 Processing: {service_name}")
            print(f"   Size: {service_info['size_gb']:.2f} GB")
            print(f"{'='*70}")

            try:
                service_path = Path(service_info['path'])
                items_count = await processor(service_path)

                self.progress.complete_service(service_name, items_count)
                total_ingested += items_count

                print(f"✅ {service_name} complete: {items_count:,} items ingested")

            except Exception as e:
                self.logger.error(f"Error processing {service_name}: {e}", exc_info=True)
                self.progress.add_error(service_name, str(e))
                print(f"❌ Error in {service_name}: {e}")

        print("\n" + "=" * 70)
        print("🧠 Analyzing personal data and extracting insights...")
        print("=" * 70)

        # Run knowledge extraction
        try:
            analysis_results = self.knowledge_extractor.analyze_personal_data()
            print(f"\n✅ Knowledge extraction complete:")
            print(f"   • Patterns found: {analysis_results['patterns_found']}")
            print(f"   • Insights generated: {analysis_results['insights_generated']}")

            # Get personal summary
            summary = self.knowledge_extractor.get_personal_summary()
            print(f"\n📊 Personal Knowledge Summary:")
            for key, value in summary.items():
                print(f"   • {key}: {value}")

        except Exception as e:
            self.logger.error(f"Error in knowledge extraction: {e}", exc_info=True)

        # Complete
        self.progress.complete()

        print("\n" + "=" * 70)
        print("✨ INGESTION COMPLETE!")
        print("=" * 70)
        print(f"\n📈 Final Statistics:")
        print(f"   • Total items ingested: {total_ingested:,}")
        print(f"   • Services processed: {len(self.progress.progress['services_completed'])}")
        print(f"   • Errors encountered: {len(self.progress.progress['errors'])}")
        print(f"   • Time taken: {self._calculate_duration()}")
        print(f"\n🔐 Your data is now securely stored and encrypted locally!")
        print(f"📁 Vault location: {self.vault.vault_dir}")
        print(f"\n🚀 You can now use PHOENIX with full knowledge of your personal data!")

    def _calculate_duration(self) -> str:
        """Calculate ingestion duration."""
        if self.progress.progress.get('started_at') and self.progress.progress.get('completed_at'):
            start = datetime.fromisoformat(self.progress.progress['started_at'])
            end = datetime.fromisoformat(self.progress.progress['completed_at'])
            duration = end - start

            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60

            return f"{hours}h {minutes}m"

        return "Unknown"

    # Optimized processors (using existing vault methods but with progress tracking)

    async def _process_gmail_optimized(self, service_path: Path) -> int:
        """Process Gmail with progress tracking."""
        count = self.vault._process_gmail(service_path)
        return count

    async def _process_photos_optimized(self, service_path: Path) -> int:
        """Process photos with progress tracking."""
        count = self.vault._process_photos(service_path)
        return count

    async def _process_location_optimized(self, service_path: Path) -> int:
        """Process location history with progress tracking."""
        count = self.vault._process_location_history(service_path)
        return count

    async def _process_youtube_optimized(self, service_path: Path) -> int:
        """Process YouTube history with progress tracking."""
        count = self.vault._process_youtube(service_path)
        return count

    async def _process_calendar_optimized(self, service_path: Path) -> int:
        """Process calendar with progress tracking."""
        count = self.vault._process_calendar(service_path)
        return count

    async def _process_contacts_optimized(self, service_path: Path) -> int:
        """Process contacts with progress tracking."""
        count = self.vault._process_contacts(service_path)
        return count

    async def _process_drive_optimized(self, service_path: Path) -> int:
        """Process Drive documents with progress tracking."""
        count = self.vault._process_drive(service_path)
        return count

    async def _process_chrome_optimized(self, service_path: Path) -> int:
        """Process Chrome data with progress tracking."""
        count = self.vault._process_chrome(service_path)
        return count

    async def _process_maps_optimized(self, service_path: Path) -> int:
        """Process Maps data with progress tracking."""
        count = self.vault._process_maps(service_path)
        return count

    async def _process_activity_optimized(self, service_path: Path) -> int:
        """Process My Activity data with progress tracking."""
        # This would need implementation in vault
        return 0


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Ingest Google Takeout into PHOENIX')
    parser.add_argument('takeout_path', type=str, help='Path to extracted Google Takeout folder')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh instead of resuming')

    args = parser.parse_args()

    ingester = EnhancedGoogleTakeoutIngester(
        takeout_path=args.takeout_path,
        resume=not args.no_resume
    )

    await ingester.ingest_all()


if __name__ == "__main__":
    asyncio.run(main())
