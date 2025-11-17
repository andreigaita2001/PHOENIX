#!/usr/bin/env python3
"""
Google Takeout Integration using google_takeout_parser library
Real, working implementation for PHOENIX data ingestion.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

try:
    from google_takeout_parser.path_dispatch import TakeoutParser
    from google_takeout_parser import models
    PARSER_AVAILABLE = True
except ImportError:
    PARSER_AVAILABLE = False
    logging.warning("google_takeout_parser not installed. Install with: pip install google-takeout-parser")


class GoogleTakeoutIntegration:
    """
    Real Google Takeout integration using the google_takeout_parser library.
    This replaces the placeholder code with actual working implementation.
    """

    def __init__(self, vault=None):
        """
        Initialize Google Takeout integration.

        Args:
            vault: PersonalDataVault instance for storing data
        """
        if not PARSER_AVAILABLE:
            raise ImportError(
                "google_takeout_parser is required. Install with:\n"
                "pip install google-takeout-parser"
            )

        self.vault = vault
        self.logger = logging.getLogger("PHOENIX.TakeoutIntegration")

        # Statistics
        self.stats = {
            'total_events': 0,
            'by_type': defaultdict(int),
            'by_service': defaultdict(int),
            'errors': []
        }

    def ingest_takeout(self, takeout_path: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Ingest Google Takeout data using the real parser.

        Args:
            takeout_path: Path to extracted Takeout directory
            use_cache: Whether to use parser cache (recommended)

        Returns:
            Ingestion results with statistics
        """
        takeout_dir = Path(takeout_path)

        if not takeout_dir.exists():
            error_msg = f"Takeout path not found: {takeout_path}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'stats': self.stats
            }

        self.logger.info(f"Starting real Google Takeout ingestion from {takeout_path}")
        self.logger.info(f"Using cache: {use_cache}")

        try:
            # Initialize the real parser
            parser = TakeoutParser(str(takeout_dir))

            # Parse all events
            self.logger.info("Parsing Takeout data (this may take a few minutes on first run)...")
            events = list(parser.parse(cache=use_cache))

            self.logger.info(f"Parsed {len(events)} total events from Takeout")

            # Process events by type
            self._process_events(events)

            return {
                'success': True,
                'total_events': len(events),
                'stats': dict(self.stats),
                'message': f"Successfully ingested {len(events)} events"
            }

        except Exception as e:
            error_msg = f"Error during ingestion: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.stats['errors'].append(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'stats': self.stats
            }

    def _process_events(self, events: List[Any]):
        """
        Process parsed events and store in vault.

        Args:
            events: List of parsed events from google_takeout_parser
        """
        for event in events:
            try:
                event_type = type(event).__name__
                self.stats['by_type'][event_type] += 1
                self.stats['total_events'] += 1

                # Process based on event type
                if isinstance(event, models.Activity):
                    self._process_activity(event)
                elif isinstance(event, models.ChromeHistory):
                    self._process_chrome_history(event)
                elif isinstance(event, models.Location):
                    self._process_location(event)
                elif isinstance(event, models.LikedYoutubeVideo):
                    self._process_youtube_like(event)
                elif isinstance(event, models.YoutubeComment):
                    self._process_youtube_comment(event)
                elif isinstance(event, models.PlayStoreAppInstall):
                    self._process_app_install(event)
                elif isinstance(event, models.SemanticLocation):
                    self._process_semantic_location(event)
                else:
                    # Generic handling for unknown types
                    self._process_generic_event(event)

                # Log progress
                if self.stats['total_events'] % 1000 == 0:
                    self.logger.info(f"Processed {self.stats['total_events']} events...")

            except Exception as e:
                self.logger.debug(f"Error processing event {event_type}: {e}")
                self.stats['errors'].append(f"{event_type}: {str(e)}")

    def _process_activity(self, activity: models.Activity):
        """Process Google My Activity events."""
        try:
            data = {
                'type': 'activity',
                'header': activity.header,
                'title': activity.title,
                'title_url': activity.titleUrl,
                'description': activity.description,
                'time': activity.time.isoformat() if activity.time else None,
                'products': activity.products,
                'subtitles': [
                    {
                        'name': sub.name,
                        'url': sub.url
                    } for sub in (activity.subtitles or [])
                ],
                'details': [
                    {
                        'name': det.name,
                        'value': det.value
                    } for det in (activity.details or [])
                ]
            }

            if self.vault:
                self.vault.store_personal_data(
                    json.dumps(data),
                    category='activity',
                    metadata={
                        'header': activity.header,
                        'title': (activity.title or '')[:100]
                    },
                    timestamp=activity.time.isoformat() if activity.time else None,
                    tags='activity,google,history'
                )

            self.stats['by_service']['MyActivity'] += 1

        except Exception as e:
            self.logger.debug(f"Error processing activity: {e}")

    def _process_chrome_history(self, entry: models.ChromeHistory):
        """Process Chrome browser history."""
        try:
            data = {
                'type': 'chrome_history',
                'url': entry.url,
                'title': entry.title,
                'time': entry.time.isoformat() if entry.time else None,
                'transition': entry.transition
            }

            # Extract domain for privacy
            domain = 'unknown'
            if entry.url and '://' in entry.url:
                try:
                    domain = entry.url.split('://')[1].split('/')[0]
                except:
                    pass

            if self.vault:
                self.vault.store_personal_data(
                    json.dumps(data),
                    category='browser_history',
                    metadata={
                        'domain': domain,
                        'title': (entry.title or '')[:100]
                    },
                    timestamp=entry.time.isoformat() if entry.time else None,
                    tags='chrome,browser,web'
                )

            self.stats['by_service']['Chrome'] += 1

        except Exception as e:
            self.logger.debug(f"Error processing Chrome history: {e}")

    def _process_location(self, location: models.Location):
        """Process location history."""
        try:
            data = {
                'type': 'location',
                'lat': location.lat,
                'lng': location.lng,
                'accuracy': location.accuracy,
                'time': location.time.isoformat() if location.time else None
            }

            if self.vault:
                self.vault.store_personal_data(
                    json.dumps(data),
                    category='location_history',
                    metadata={
                        'lat': location.lat,
                        'lng': location.lng
                    },
                    timestamp=location.time.isoformat() if location.time else None,
                    tags='location,gps,travel'
                )

            self.stats['by_service']['LocationHistory'] += 1

        except Exception as e:
            self.logger.debug(f"Error processing location: {e}")

    def _process_youtube_like(self, like: models.LikedYoutubeVideo):
        """Process YouTube liked videos."""
        try:
            data = {
                'type': 'youtube_like',
                'title': like.title,
                'url': like.url,
                'time': like.time.isoformat() if like.time else None
            }

            if self.vault:
                self.vault.store_personal_data(
                    json.dumps(data),
                    category='youtube_history',
                    metadata={
                        'title': (like.title or '')[:100],
                        'type': 'like'
                    },
                    timestamp=like.time.isoformat() if like.time else None,
                    tags='youtube,video,like'
                )

            self.stats['by_service']['YouTube'] += 1

        except Exception as e:
            self.logger.debug(f"Error processing YouTube like: {e}")

    def _process_youtube_comment(self, comment: models.YoutubeComment):
        """Process YouTube comments."""
        try:
            data = {
                'type': 'youtube_comment',
                'content': comment.content,
                'time': comment.time.isoformat() if comment.time else None
            }

            if self.vault:
                self.vault.store_personal_data(
                    json.dumps(data),
                    category='youtube_history',
                    metadata={
                        'content': (comment.content or '')[:100],
                        'type': 'comment'
                    },
                    timestamp=comment.time.isoformat() if comment.time else None,
                    tags='youtube,comment,interaction'
                )

            self.stats['by_service']['YouTube'] += 1

        except Exception as e:
            self.logger.debug(f"Error processing YouTube comment: {e}")

    def _process_app_install(self, app: models.PlayStoreAppInstall):
        """Process Play Store app installations."""
        try:
            data = {
                'type': 'app_install',
                'title': app.title,
                'time': app.time.isoformat() if app.time else None
            }

            if self.vault:
                self.vault.store_personal_data(
                    json.dumps(data),
                    category='app_data',
                    metadata={
                        'app': (app.title or '')[:100]
                    },
                    timestamp=app.time.isoformat() if app.time else None,
                    tags='apps,playstore,android'
                )

            self.stats['by_service']['PlayStore'] += 1

        except Exception as e:
            self.logger.debug(f"Error processing app install: {e}")

    def _process_semantic_location(self, location: models.SemanticLocation):
        """Process semantic location (places visited)."""
        try:
            data = {
                'type': 'semantic_location',
                'start': location.start.isoformat() if location.start else None,
                'end': location.end.isoformat() if location.end else None,
                'location_type': str(type(location).__name__)
            }

            if self.vault:
                self.vault.store_personal_data(
                    json.dumps(data),
                    category='location_history',
                    metadata={
                        'type': 'semantic'
                    },
                    timestamp=location.start.isoformat() if location.start else None,
                    tags='location,places,semantic'
                )

            self.stats['by_service']['LocationHistory'] += 1

        except Exception as e:
            self.logger.debug(f"Error processing semantic location: {e}")

    def _process_generic_event(self, event: Any):
        """Process generic events."""
        try:
            event_type = type(event).__name__

            # Try to extract basic info
            data = {
                'type': event_type,
                'raw': str(event)[:500]  # Limit size
            }

            # Try to get timestamp
            timestamp = None
            if hasattr(event, 'time'):
                timestamp = event.time.isoformat() if event.time else None

            if self.vault:
                self.vault.store_personal_data(
                    json.dumps(data),
                    category='other',
                    metadata={
                        'event_type': event_type
                    },
                    timestamp=timestamp,
                    tags=f'generic,{event_type.lower()}'
                )

            self.stats['by_service']['Other'] += 1

        except Exception as e:
            self.logger.debug(f"Error processing generic event: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {
            'total_events': self.stats['total_events'],
            'events_by_type': dict(self.stats['by_type']),
            'events_by_service': dict(self.stats['by_service']),
            'errors': len(self.stats['errors']),
            'error_details': self.stats['errors'][:10]  # First 10 errors
        }

    def clear_cache(self):
        """Clear the parser cache (useful for testing)."""
        try:
            from google_takeout_parser.cache import clear_cache
            clear_cache()
            self.logger.info("Parser cache cleared")
        except Exception as e:
            self.logger.warning(f"Could not clear cache: {e}")
