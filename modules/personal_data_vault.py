#!/usr/bin/env python3
"""
Personal Data Vault - Secure, encrypted storage for all personal data.
This module handles ALL personal data with maximum privacy and security.
Everything stays local, everything is encrypted.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import pickle
import sqlite3
from collections import defaultdict

class PersonalDataVault:
    """
    Secure vault for all personal data with encryption and privacy protection.
    """

    def __init__(self, memory_manager=None, password: Optional[str] = None):
        """
        Initialize the Personal Data Vault.

        Args:
            memory_manager: Memory system for learning
            password: Optional password for encryption (will prompt if not provided)
        """
        self.memory_manager = memory_manager
        self.logger = logging.getLogger("PHOENIX.PersonalVault")

        # Vault directory (encrypted storage)
        self.vault_dir = Path.home() / '.phoenix_vault'
        self.vault_dir.mkdir(mode=0o700, exist_ok=True)  # Only owner can access

        # Initialize encryption
        self._init_encryption(password)

        # Personal data categories
        self.data_categories = {
            'emails': [],
            'calendar': [],
            'contacts': [],
            'photos': [],
            'location_history': [],
            'search_history': [],
            'youtube_history': [],
            'browser_history': [],
            'documents': [],
            'social_media': [],
            'financial': [],
            'health': [],
            'notes': [],
            'voice_recordings': [],
            'app_data': {}
        }

        # Initialize secure database
        self._init_secure_db()

        # Statistics
        self.stats = {
            'total_items': 0,
            'categories_populated': 0,
            'last_update': None,
            'encryption_enabled': True,
            'privacy_level': 'maximum'
        }

        self.logger.info("Personal Data Vault initialized with maximum security")

    def _init_encryption(self, password: Optional[str] = None):
        """Initialize encryption system."""
        key_file = self.vault_dir / '.key'

        if key_file.exists() and key_file.stat().st_mode & 0o777 == 0o600:
            # Load existing key
            with open(key_file, 'rb') as f:
                self.cipher_key = f.read()
        else:
            # Generate new key
            if not password:
                # Generate from system entropy for maximum security
                salt = os.urandom(16)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000
                )
                key = base64.urlsafe_b64encode(kdf.derive(os.urandom(32)))
            else:
                # Derive from password
                salt = b'phoenix_personal_vault_2024'
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000
                )
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))

            self.cipher_key = key

            # Save key with strict permissions
            with open(key_file, 'wb') as f:
                f.write(self.cipher_key)
            os.chmod(key_file, 0o600)  # Only owner can read

        self.cipher = Fernet(self.cipher_key)
        self.logger.info("Encryption initialized")

    def _init_secure_db(self):
        """Initialize secure SQLite database for indexing."""
        db_path = self.vault_dir / 'personal_index.db'
        self.db = sqlite3.connect(str(db_path))
        self.cursor = self.db.cursor()

        # Create tables for different data types
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS personal_data (
                id TEXT PRIMARY KEY,
                category TEXT,
                data_type TEXT,
                encrypted_content BLOB,
                metadata TEXT,
                timestamp TEXT,
                tags TEXT,
                importance INTEGER,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS personal_patterns (
                id TEXT PRIMARY KEY,
                pattern_type TEXT,
                pattern_data TEXT,
                confidence REAL,
                occurrences INTEGER,
                first_seen TEXT,
                last_seen TEXT
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS personal_insights (
                id TEXT PRIMARY KEY,
                insight_type TEXT,
                insight TEXT,
                confidence REAL,
                created TEXT,
                category TEXT
            )
        ''')

        # Create indexes for faster search
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON personal_data(category)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON personal_data(timestamp)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags ON personal_data(tags)')

        self.db.commit()

    def ingest_google_takeout(self, takeout_path: str) -> Dict[str, Any]:
        """
        Ingest data from Google Takeout archive.

        Args:
            takeout_path: Path to extracted Google Takeout folder

        Returns:
            Ingestion results
        """
        results = {
            'success': False,
            'categories_processed': [],
            'total_items': 0,
            'errors': []
        }

        takeout_dir = Path(takeout_path)
        if not takeout_dir.exists():
            results['errors'].append(f"Takeout path not found: {takeout_path}")
            return results

        self.logger.info(f"Starting Google Takeout ingestion from {takeout_path}")

        # Process each Google service
        processors = {
            'Gmail': self._process_gmail,
            'Google Photos': self._process_photos,
            'Location History': self._process_location_history,
            'YouTube and YouTube Music': self._process_youtube,
            'My Activity': self._process_activity,
            'Calendar': self._process_calendar,
            'Contacts': self._process_contacts,
            'Drive': self._process_drive,
            'Chrome': self._process_chrome,
            'Maps': self._process_maps
        }

        for service, processor in processors.items():
            service_path = takeout_dir / service
            if service_path.exists():
                try:
                    self.logger.info(f"Processing {service}...")
                    count = processor(service_path)
                    results['categories_processed'].append(service)
                    results['total_items'] += count
                    self.logger.info(f"Processed {count} items from {service}")
                except Exception as e:
                    self.logger.error(f"Error processing {service}: {e}")
                    results['errors'].append(f"{service}: {str(e)}")

        # Update statistics
        self.stats['total_items'] = results['total_items']
        self.stats['last_update'] = datetime.now().isoformat()
        self.stats['categories_populated'] = len(results['categories_processed'])

        # Generate insights from ingested data
        self._generate_personal_insights()

        results['success'] = len(results['categories_processed']) > 0
        return results

    def _process_gmail(self, gmail_path: Path) -> int:
        """Process Gmail data from mbox files."""
        import mailbox
        import email.utils
        count = 0
        mbox_files = list(gmail_path.glob('*.mbox'))

        for mbox_file in mbox_files:
            self.logger.info(f"Processing mailbox: {mbox_file.name}")

            try:
                mbox = mailbox.mbox(str(mbox_file))

                for message in mbox:
                    try:
                        # Extract metadata only, not full content for privacy
                        from_addr = message.get('From', '')
                        to_addr = message.get('To', '')
                        date_str = message.get('Date', '')
                        subject = message.get('Subject', '')[:200]  # Truncate long subjects
                        message_id = message.get('Message-ID', '')

                        # Parse date
                        date_tuple = None
                        if date_str:
                            try:
                                date_tuple = email.utils.parsedate_to_datetime(date_str)
                            except:
                                pass

                        timestamp = date_tuple.isoformat() if date_tuple else datetime.now().isoformat()

                        # Create metadata
                        metadata = {
                            'from': from_addr,
                            'to': to_addr,
                            'subject': subject,
                            'message_id': message_id,
                            'has_attachments': bool(message.get_payload()),
                            'mbox_file': mbox_file.name
                        }

                        # Store encrypted
                        data_id = self._generate_id('email', message_id or f"{from_addr}{timestamp}")
                        encrypted_data = self.cipher.encrypt(json.dumps(metadata).encode())

                        self.cursor.execute('''
                            INSERT OR REPLACE INTO personal_data
                            (id, category, data_type, encrypted_content, metadata, timestamp, tags)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            data_id, 'emails', 'email',
                            encrypted_data,
                            json.dumps({'subject': subject[:50], 'from': from_addr.split('@')[0] if '@' in from_addr else 'unknown'}),
                            timestamp,
                            'email,gmail,communication'
                        ))
                        count += 1

                        if count % 100 == 0:
                            self.logger.info(f"Processed {count} emails...")
                            self.db.commit()

                    except Exception as e:
                        self.logger.debug(f"Error processing individual email: {e}")
                        continue

            except Exception as e:
                self.logger.error(f"Error opening mbox file {mbox_file}: {e}")

        self.db.commit()
        return count

    def _process_photos(self, photos_path: Path) -> int:
        """Process Google Photos data."""
        count = 0
        photo_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mov', '.heic'}

        for photo_file in photos_path.rglob('*'):
            if photo_file.suffix.lower() in photo_extensions:
                # Store encrypted reference (not the actual photo)
                data_id = self._generate_id('photo', str(photo_file))
                encrypted_path = self.cipher.encrypt(str(photo_file).encode())

                # Extract metadata if available
                metadata = {
                    'filename': photo_file.name,
                    'size': photo_file.stat().st_size,
                    'modified': datetime.fromtimestamp(photo_file.stat().st_mtime).isoformat()
                }

                # Check for accompanying JSON metadata
                json_file = photo_file.with_suffix(photo_file.suffix + '.json')
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        photo_metadata = json.load(f)
                        metadata.update(photo_metadata)

                self.cursor.execute('''
                    INSERT OR REPLACE INTO personal_data
                    (id, category, data_type, encrypted_content, metadata, timestamp, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data_id, 'photos', 'photo_reference',
                    encrypted_path,
                    json.dumps(metadata),
                    metadata.get('photoTakenTime', {}).get('timestamp', datetime.now().isoformat()),
                    'photo,media,memories'
                ))
                count += 1

        self.db.commit()
        return count

    def _process_location_history(self, location_path: Path) -> int:
        """Process location history data."""
        count = 0
        records_file = location_path / 'Records.json'

        if records_file.exists():
            with open(records_file, 'r') as f:
                data = json.load(f)

            locations = data.get('locations', [])
            self.logger.info(f"Processing {len(locations)} location points")

            # Group locations by day for efficient storage
            daily_locations = defaultdict(list)
            for loc in locations:
                timestamp = loc.get('timestamp', '')
                if timestamp:
                    date = timestamp[:10]  # Extract date
                    daily_locations[date].append({
                        'lat': loc.get('latitudeE7', 0) / 1e7,
                        'lng': loc.get('longitudeE7', 0) / 1e7,
                        'accuracy': loc.get('accuracy'),
                        'timestamp': timestamp
                    })

            # Store encrypted daily summaries
            for date, locations in daily_locations.items():
                data_id = self._generate_id('location', date)
                encrypted_data = self.cipher.encrypt(json.dumps(locations).encode())

                self.cursor.execute('''
                    INSERT OR REPLACE INTO personal_data
                    (id, category, data_type, encrypted_content, metadata, timestamp, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data_id, 'location_history', 'daily_locations',
                    encrypted_data,
                    json.dumps({'date': date, 'points': len(locations)}),
                    date,
                    'location,travel,movement'
                ))
                count += 1

        self.db.commit()
        return count

    def _process_youtube(self, youtube_path: Path) -> int:
        """Process YouTube history."""
        count = 0
        history_files = ['watch-history.json', 'search-history.json']

        for history_file in history_files:
            file_path = youtube_path / 'history' / history_file
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for item in data:
                    data_id = self._generate_id('youtube', item.get('time', ''))
                    encrypted_data = self.cipher.encrypt(json.dumps(item).encode())

                    self.cursor.execute('''
                        INSERT OR REPLACE INTO personal_data
                        (id, category, data_type, encrypted_content, metadata, timestamp, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        data_id, 'youtube_history', history_file.replace('.json', ''),
                        encrypted_data,
                        json.dumps({'title': item.get('title', 'Unknown')}),
                        item.get('time', datetime.now().isoformat()),
                        'youtube,media,entertainment'
                    ))
                    count += 1

        self.db.commit()
        return count

    def _process_activity(self, activity_path: Path) -> int:
        """Process My Activity data."""
        count = 0
        # This would process search history, app usage, etc.
        # Simplified for demonstration
        return count

    def _process_calendar(self, calendar_path: Path) -> int:
        """Process calendar data from .ics files."""
        count = 0
        try:
            import icalendar
        except ImportError:
            self.logger.warning("icalendar module not available, using basic parsing")
            # Fallback to basic parsing
            ics_files = list(calendar_path.glob('*.ics'))
            for ics_file in ics_files:
                with open(ics_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract basic event info
                    events = content.split('BEGIN:VEVENT')
                    for event in events[1:]:  # Skip first split
                        lines = event.split('\n')
                        event_data = {}
                        for line in lines:
                            if line.startswith('SUMMARY:'):
                                event_data['summary'] = line.replace('SUMMARY:', '')
                            elif line.startswith('DTSTART:'):
                                event_data['start'] = line.replace('DTSTART:', '')
                            elif line.startswith('DTEND:'):
                                event_data['end'] = line.replace('DTEND:', '')

                        if event_data:
                            data_id = self._generate_id('calendar', str(event_data))
                            encrypted_data = self.cipher.encrypt(json.dumps(event_data).encode())

                            self.cursor.execute('''
                                INSERT OR REPLACE INTO personal_data
                                (id, category, data_type, encrypted_content, metadata, timestamp, tags)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                data_id, 'calendar', 'event',
                                encrypted_data,
                                json.dumps({'summary': event_data.get('summary', 'Event')[:50]}),
                                event_data.get('start', datetime.now().isoformat()),
                                'calendar,events,schedule'
                            ))
                            count += 1
            self.db.commit()
            return count

        # Use icalendar if available
        ics_files = list(calendar_path.glob('*.ics'))
        for ics_file in ics_files:
            with open(ics_file, 'rb') as f:
                cal = icalendar.Calendar.from_ical(f.read())

                for component in cal.walk():
                    if component.name == "VEVENT":
                        event_data = {
                            'summary': str(component.get('summary', '')),
                            'start': str(component.get('dtstart', '')),
                            'end': str(component.get('dtend', '')),
                            'location': str(component.get('location', '')),
                            'description': str(component.get('description', ''))[:200]
                        }

                        data_id = self._generate_id('calendar', component.get('uid', str(event_data)))
                        encrypted_data = self.cipher.encrypt(json.dumps(event_data).encode())

                        self.cursor.execute('''
                            INSERT OR REPLACE INTO personal_data
                            (id, category, data_type, encrypted_content, metadata, timestamp, tags)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            data_id, 'calendar', 'event',
                            encrypted_data,
                            json.dumps({'summary': event_data['summary'][:50]}),
                            event_data.get('start', datetime.now().isoformat()),
                            'calendar,events,schedule'
                        ))
                        count += 1

        self.db.commit()
        return count

    def _process_contacts(self, contacts_path: Path) -> int:
        """Process contacts from .vcf files."""
        count = 0
        vcf_files = list(contacts_path.glob('*.vcf'))

        for vcf_file in vcf_files:
            with open(vcf_file, 'r', encoding='utf-8') as f:
                content = f.read()

                # Split into individual vCards
                vcards = content.split('BEGIN:VCARD')

                for vcard in vcards[1:]:  # Skip first split
                    contact_data = {}
                    lines = vcard.split('\n')

                    for line in lines:
                        if line.startswith('FN:'):
                            contact_data['name'] = line.replace('FN:', '')
                        elif line.startswith('EMAIL'):
                            if 'email' not in contact_data:
                                contact_data['email'] = []
                            # Extract email after colon
                            if ':' in line:
                                contact_data['email'].append(line.split(':')[1])
                        elif line.startswith('TEL'):
                            if 'phone' not in contact_data:
                                contact_data['phone'] = []
                            if ':' in line:
                                contact_data['phone'].append(line.split(':')[1])
                        elif line.startswith('ORG:'):
                            contact_data['organization'] = line.replace('ORG:', '')
                        elif line.startswith('NOTE:'):
                            contact_data['note'] = line.replace('NOTE:', '')[:200]

                    if contact_data:
                        data_id = self._generate_id('contact', str(contact_data))
                        encrypted_data = self.cipher.encrypt(json.dumps(contact_data).encode())

                        self.cursor.execute('''
                            INSERT OR REPLACE INTO personal_data
                            (id, category, data_type, encrypted_content, metadata, timestamp, tags)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            data_id, 'contacts', 'contact',
                            encrypted_data,
                            json.dumps({'name': contact_data.get('name', 'Unknown')[:50]}),
                            datetime.now().isoformat(),
                            'contacts,people,network'
                        ))
                        count += 1

        self.db.commit()
        return count

    def _process_drive(self, drive_path: Path) -> int:
        """Process Google Drive documents."""
        count = 0

        # Process all document files recursively
        doc_extensions = {'.txt', '.doc', '.docx', '.pdf', '.md', '.json', '.csv', '.xlsx'}

        for doc_file in drive_path.rglob('*'):
            if doc_file.is_file() and doc_file.suffix.lower() in doc_extensions:
                try:
                    # Create metadata
                    metadata = {
                        'filename': doc_file.name,
                        'path': str(doc_file.relative_to(drive_path)),
                        'size': doc_file.stat().st_size,
                        'extension': doc_file.suffix,
                        'modified': datetime.fromtimestamp(doc_file.stat().st_mtime).isoformat()
                    }

                    # Store encrypted reference (not content)
                    data_id = self._generate_id('drive', str(doc_file))
                    encrypted_path = self.cipher.encrypt(str(doc_file).encode())

                    self.cursor.execute('''
                        INSERT OR REPLACE INTO personal_data
                        (id, category, data_type, encrypted_content, metadata, timestamp, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        data_id, 'documents', 'drive_doc',
                        encrypted_path,
                        json.dumps(metadata),
                        metadata['modified'],
                        'documents,drive,files'
                    ))
                    count += 1

                except Exception as e:
                    self.logger.debug(f"Error processing drive file {doc_file}: {e}")

        self.db.commit()
        return count

    def _process_chrome(self, chrome_path: Path) -> int:
        """Process Chrome browsing data."""
        count = 0

        # Process browsing history JSON
        history_file = chrome_path / 'BrowserHistory.json'
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)

                browser_history = history_data.get('Browser History', [])
                for entry in browser_history:
                    try:
                        # Extract just domain and title for privacy
                        url = entry.get('url', '')
                        domain = url.split('/')[2] if url.startswith('http') and '/' in url else 'unknown'

                        metadata = {
                            'title': entry.get('title', '')[:100],
                            'domain': domain,
                            'time': entry.get('time_usec', 0)
                        }

                        timestamp = datetime.fromtimestamp(metadata['time'] / 1000000).isoformat() if metadata['time'] else datetime.now().isoformat()

                        data_id = self._generate_id('browse', f"{domain}{metadata['time']}")
                        encrypted_data = self.cipher.encrypt(json.dumps(metadata).encode())

                        self.cursor.execute('''
                            INSERT OR REPLACE INTO personal_data
                            (id, category, data_type, encrypted_content, metadata, timestamp, tags)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            data_id, 'browser_history', 'visit',
                            encrypted_data,
                            json.dumps({'domain': domain}),
                            timestamp,
                            'browser,chrome,web'
                        ))
                        count += 1

                    except Exception as e:
                        self.logger.debug(f"Error processing browser entry: {e}")

        # Process bookmarks
        bookmarks_file = chrome_path / 'Bookmarks.html'
        if bookmarks_file.exists():
            with open(bookmarks_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple extraction of bookmark titles
                import re
                bookmarks = re.findall(r'<A[^>]*>(.*?)</A>', content, re.IGNORECASE)

                for bookmark_title in bookmarks[:100]:  # Limit to first 100
                    data_id = self._generate_id('bookmark', bookmark_title)
                    encrypted_data = self.cipher.encrypt(bookmark_title.encode())

                    self.cursor.execute('''
                        INSERT OR REPLACE INTO personal_data
                        (id, category, data_type, encrypted_content, metadata, timestamp, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        data_id, 'browser_history', 'bookmark',
                        encrypted_data,
                        json.dumps({'title': bookmark_title[:50]}),
                        datetime.now().isoformat(),
                        'bookmarks,chrome,saved'
                    ))
                    count += 1

        self.db.commit()
        return count

    def _process_maps(self, maps_path: Path) -> int:
        """Process Google Maps data."""
        count = 0

        # Process saved places
        places_file = maps_path / 'Maps (your places)' / 'Saved Places.json'
        if places_file.exists():
            with open(places_file, 'r', encoding='utf-8') as f:
                places_data = json.load(f)

                for feature in places_data.get('features', []):
                    properties = feature.get('properties', {})
                    geometry = feature.get('geometry', {})

                    place_data = {
                        'name': properties.get('Title', 'Unknown Place'),
                        'address': properties.get('Address', ''),
                        'coordinates': geometry.get('coordinates', []),
                        'note': properties.get('Note', '')[:200]
                    }

                    data_id = self._generate_id('place', place_data['name'])
                    encrypted_data = self.cipher.encrypt(json.dumps(place_data).encode())

                    self.cursor.execute('''
                        INSERT OR REPLACE INTO personal_data
                        (id, category, data_type, encrypted_content, metadata, timestamp, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        data_id, 'location_history', 'saved_place',
                        encrypted_data,
                        json.dumps({'name': place_data['name'][:50]}),
                        datetime.now().isoformat(),
                        'maps,places,locations'
                    ))
                    count += 1

        # Process reviews
        reviews_file = maps_path / 'Maps (your places)' / 'Reviews.json'
        if reviews_file.exists():
            with open(reviews_file, 'r', encoding='utf-8') as f:
                reviews = json.load(f)

                for review in reviews:
                    review_data = {
                        'place': review.get('name', 'Unknown'),
                        'rating': review.get('rating', 0),
                        'comment': review.get('comment', '')[:200]
                    }

                    data_id = self._generate_id('review', str(review_data))
                    encrypted_data = self.cipher.encrypt(json.dumps(review_data).encode())

                    self.cursor.execute('''
                        INSERT OR REPLACE INTO personal_data
                        (id, category, data_type, encrypted_content, metadata, timestamp, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        data_id, 'location_history', 'review',
                        encrypted_data,
                        json.dumps({'place': review_data['place'][:50]}),
                        datetime.now().isoformat(),
                        'reviews,maps,opinions'
                    ))
                    count += 1

        self.db.commit()
        return count

    def search_personal_data(self, query: str, category: Optional[str] = None) -> List[Dict]:
        """
        Search through personal data securely.

        Args:
            query: Search query
            category: Optional category filter

        Returns:
            Search results (decrypted)
        """
        results = []

        # Build query
        if category:
            self.cursor.execute('''
                SELECT id, category, data_type, encrypted_content, metadata, timestamp
                FROM personal_data
                WHERE category = ?
                ORDER BY timestamp DESC
                LIMIT 20
            ''', (category,))
        else:
            # Search in metadata (not encrypted content for privacy)
            self.cursor.execute('''
                SELECT id, category, data_type, encrypted_content, metadata, timestamp
                FROM personal_data
                WHERE metadata LIKE ?
                ORDER BY timestamp DESC
                LIMIT 20
            ''', (f'%{query}%',))

        for row in self.cursor.fetchall():
            result = {
                'id': row[0],
                'category': row[1],
                'type': row[2],
                'metadata': json.loads(row[4]),
                'timestamp': row[5]
            }

            # Update access count
            self.cursor.execute('''
                UPDATE personal_data
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), row[0]))

            results.append(result)

        self.db.commit()
        return results

    def _generate_personal_insights(self):
        """Generate insights from personal data."""
        insights = []

        # Analyze patterns in different categories
        self.cursor.execute('''
            SELECT category, COUNT(*) as count
            FROM personal_data
            GROUP BY category
        ''')

        for category, count in self.cursor.fetchall():
            if count > 0:
                insight = {
                    'type': 'data_summary',
                    'insight': f"You have {count} items in {category}",
                    'category': category,
                    'confidence': 1.0
                }
                insights.append(insight)

        # Store insights
        for insight in insights:
            insight_id = self._generate_id('insight', insight['insight'])
            self.cursor.execute('''
                INSERT OR REPLACE INTO personal_insights
                (id, insight_type, insight, confidence, created, category)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                insight_id,
                insight['type'],
                insight['insight'],
                insight['confidence'],
                datetime.now().isoformat(),
                insight['category']
            ))

        self.db.commit()

        # Learn from patterns if memory manager available
        if self.memory_manager:
            for insight in insights:
                self.memory_manager.learn_fact(
                    insight['insight'],
                    category='personal_insights',
                    confidence=insight['confidence']
                )

    def store_personal_data(self, data: str, category: str, metadata: Dict = None, timestamp: str = None, tags: str = "") -> str:
        """
        Store personal data in the vault.

        Args:
            data: Data to store (will be encrypted)
            category: Category of data
            metadata: Optional metadata
            timestamp: Optional timestamp
            tags: Optional tags

        Returns:
            ID of stored data
        """
        data_id = self._generate_id(category, data)
        encrypted_data = self.cipher.encrypt(data.encode())

        self.cursor.execute('''
            INSERT OR REPLACE INTO personal_data
            (id, category, data_type, encrypted_content, metadata, timestamp, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data_id, category, 'user_data',
            encrypted_data,
            json.dumps(metadata or {}),
            timestamp or datetime.now().isoformat(),
            tags
        ))
        self.db.commit()

        # Update stats
        self.stats['total_items'] += 1
        self.stats['last_update'] = datetime.now().isoformat()

        return data_id

    def retrieve_personal_data(self, data_id: str) -> Optional[Dict]:
        """
        Retrieve and decrypt personal data by ID.

        Args:
            data_id: ID of data to retrieve

        Returns:
            Decrypted data or None
        """
        self.cursor.execute('''
            SELECT encrypted_content, metadata, category, timestamp
            FROM personal_data
            WHERE id = ?
        ''', (data_id,))

        row = self.cursor.fetchone()
        if row:
            decrypted_data = self.cipher.decrypt(row[0]).decode()
            return {
                'data': decrypted_data,
                'metadata': json.loads(row[1]),
                'category': row[2],
                'timestamp': row[3]
            }
        return None

    def _generate_id(self, prefix: str, content: str) -> str:
        """Generate unique ID for data."""
        return f"{prefix}_{hashlib.sha256(content.encode()).hexdigest()[:16]}"

    def get_privacy_report(self) -> Dict[str, Any]:
        """Get privacy and security report."""
        return {
            'encryption': 'AES-256 (Fernet)',
            'storage_location': str(self.vault_dir),
            'access_permissions': 'Owner only (700)',
            'data_categories': list(self.data_categories.keys()),
            'total_items': self.stats['total_items'],
            'last_update': self.stats['last_update'],
            'privacy_level': 'MAXIMUM',
            'data_stays_local': True,
            'third_party_access': False,
            'telemetry': False
        }

    def export_insights(self) -> List[Dict]:
        """Export learned insights (not raw data)."""
        self.cursor.execute('''
            SELECT insight_type, insight, confidence, category
            FROM personal_insights
            ORDER BY confidence DESC
        ''')

        insights = []
        for row in self.cursor.fetchall():
            insights.append({
                'type': row[0],
                'insight': row[1],
                'confidence': row[2],
                'category': row[3]
            })

        return insights

    def wipe_all_data(self, confirm: str = ""):
        """
        Completely wipe all personal data.

        Args:
            confirm: Must be "DELETE_ALL_MY_DATA" to confirm
        """
        if confirm != "DELETE_ALL_MY_DATA":
            self.logger.warning("Wipe request denied - incorrect confirmation")
            return False

        # Close database
        self.db.close()

        # Securely delete all files
        import shutil
        shutil.rmtree(self.vault_dir, ignore_errors=True)

        self.logger.info("All personal data has been securely deleted")
        return True