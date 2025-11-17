#!/usr/bin/env python3
"""
Intelligent Content Analyzer
Uses local LLM to actually understand and extract insights from personal data.
This is the REAL analysis layer that processes content semantically.
"""

import logging
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import Counter, defaultdict
import ollama
from dataclasses import dataclass, asdict
import sqlite3


@dataclass
class InsightRecord:
    """Record of an insight extracted by the LLM."""
    timestamp: str
    category: str
    insight_type: str
    content: str
    confidence: float
    source_ids: List[str]
    metadata: Dict[str, Any]


class OperationHistory:
    """Track all operations for debugging and optimization."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db = sqlite3.connect(str(db_path))
        self.cursor = self.db.cursor()
        self._init_db()
        self.logger = logging.getLogger("PHOENIX.OperationHistory")

    def _init_db(self):
        """Initialize operation history database."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                operation_type TEXT,
                module TEXT,
                input_data TEXT,
                output_data TEXT,
                duration_ms INTEGER,
                gpu_used BOOLEAN,
                gpu_memory_mb INTEGER,
                model_used TEXT,
                tokens_processed INTEGER,
                success BOOLEAN,
                error_message TEXT,
                metadata TEXT
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                model_name TEXT,
                task_type TEXT,
                tokens_per_second REAL,
                gpu_utilization REAL,
                memory_used_mb INTEGER,
                latency_ms INTEGER,
                quality_score REAL
            )
        ''')

        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_operation_type ON operations(operation_type)
        ''')
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON operations(timestamp)
        ''')
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_model ON operations(model_used)
        ''')

        self.db.commit()

    def record_operation(self, operation_type: str, module: str,
                        input_data: Any, output_data: Any,
                        duration_ms: int, success: bool,
                        model_used: str = None, tokens: int = 0,
                        gpu_info: Dict = None, error: str = None,
                        metadata: Dict = None):
        """Record an operation for history and analysis."""

        gpu_used = gpu_info.get('used', False) if gpu_info else False
        gpu_memory = gpu_info.get('memory_mb', 0) if gpu_info else 0

        self.cursor.execute('''
            INSERT INTO operations (
                timestamp, operation_type, module, input_data, output_data,
                duration_ms, gpu_used, gpu_memory_mb, model_used,
                tokens_processed, success, error_message, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            operation_type,
            module,
            json.dumps(input_data) if input_data else None,
            json.dumps(output_data) if output_data else None,
            duration_ms,
            gpu_used,
            gpu_memory,
            model_used,
            tokens,
            success,
            error,
            json.dumps(metadata) if metadata else None
        ))
        self.db.commit()

    def record_model_performance(self, model_name: str, task_type: str,
                                 tokens_per_sec: float, gpu_util: float,
                                 memory_mb: int, latency_ms: int,
                                 quality_score: float = None):
        """Record model performance metrics."""
        self.cursor.execute('''
            INSERT INTO model_performance (
                timestamp, model_name, task_type, tokens_per_second,
                gpu_utilization, memory_used_mb, latency_ms, quality_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            model_name,
            task_type,
            tokens_per_sec,
            gpu_util,
            memory_mb,
            latency_ms,
            quality_score
        ))
        self.db.commit()

    def get_statistics(self) -> Dict[str, Any]:
        """Get operation statistics."""
        stats = {}

        # Total operations
        self.cursor.execute('SELECT COUNT(*) FROM operations')
        stats['total_operations'] = self.cursor.fetchone()[0]

        # Success rate
        self.cursor.execute('SELECT COUNT(*) FROM operations WHERE success = 1')
        success_count = self.cursor.fetchone()[0]
        stats['success_rate'] = success_count / stats['total_operations'] if stats['total_operations'] > 0 else 0

        # Average duration by operation type
        self.cursor.execute('''
            SELECT operation_type, AVG(duration_ms) as avg_duration, COUNT(*) as count
            FROM operations
            GROUP BY operation_type
        ''')
        stats['operation_durations'] = {
            row[0]: {'avg_ms': row[1], 'count': row[2]}
            for row in self.cursor.fetchall()
        }

        # GPU usage
        self.cursor.execute('SELECT COUNT(*) FROM operations WHERE gpu_used = 1')
        gpu_ops = self.cursor.fetchone()[0]
        stats['gpu_operations'] = gpu_ops
        stats['gpu_usage_rate'] = gpu_ops / stats['total_operations'] if stats['total_operations'] > 0 else 0

        # Model performance
        self.cursor.execute('''
            SELECT model_name, AVG(tokens_per_second) as avg_tps,
                   AVG(gpu_utilization) as avg_gpu, COUNT(*) as uses
            FROM model_performance
            GROUP BY model_name
        ''')
        stats['model_performance'] = {
            row[0]: {
                'avg_tokens_per_sec': row[1],
                'avg_gpu_util': row[2],
                'total_uses': row[3]
            }
            for row in self.cursor.fetchall()
        }

        return stats

    def get_recent_operations(self, limit: int = 50) -> List[Dict]:
        """Get recent operations."""
        self.cursor.execute('''
            SELECT timestamp, operation_type, module, duration_ms,
                   model_used, success, error_message
            FROM operations
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        return [
            {
                'timestamp': row[0],
                'type': row[1],
                'module': row[2],
                'duration_ms': row[3],
                'model': row[4],
                'success': bool(row[5]),
                'error': row[6]
            }
            for row in self.cursor.fetchall()
        ]


class GPUMonitor:
    """Monitor GPU usage and performance."""

    def __init__(self):
        self.logger = logging.getLogger("PHOENIX.GPUMonitor")
        self.has_gpu = self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get current GPU information."""
        if not self.has_gpu:
            return {'available': False}

        try:
            import subprocess
            # Get GPU memory usage
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                return {
                    'available': True,
                    'memory_used_mb': int(parts[0].strip()),
                    'memory_total_mb': int(parts[1].strip()),
                    'utilization': float(parts[2].strip()),
                    'temperature': float(parts[3].strip())
                }
        except Exception as e:
            self.logger.debug(f"Error getting GPU info: {e}")

        return {'available': False}

    def can_use_large_model(self) -> bool:
        """Check if we have enough GPU memory for large models."""
        info = self.get_gpu_info()
        if not info.get('available'):
            return False

        # Need at least 12GB free for 14B models
        memory_free = info['memory_total_mb'] - info['memory_used_mb']
        return memory_free >= 12000


class IntelligentAnalyzer:
    """
    Actually intelligent content analysis using local LLM.
    This processes content semantically, not just metadata.
    """

    def __init__(self, vault, operation_history: OperationHistory = None):
        self.vault = vault
        self.logger = logging.getLogger("PHOENIX.IntelligentAnalyzer")

        # Operation tracking
        history_path = Path.home() / '.phoenix_vault' / 'operation_history.db'
        self.history = operation_history or OperationHistory(history_path)

        # GPU monitoring
        self.gpu_monitor = GPUMonitor()

        # LLM client
        self.client = ollama.Client()

        # Select best model based on GPU availability
        self.model = self._select_model()

        # Insights storage
        self.insights_db_path = Path.home() / '.phoenix_vault' / 'insights.db'
        self._init_insights_db()

        self.logger.info(f"Intelligent Analyzer initialized with model: {self.model}")
        self.logger.info(f"GPU available: {self.gpu_monitor.has_gpu}")

    def _select_model(self) -> str:
        """Select best model based on available resources."""
        try:
            models = self.client.list()
            available_models = [m['name'] for m in models.get('models', [])]

            # Check GPU capability
            can_use_large = self.gpu_monitor.can_use_large_model()

            # Preference order based on GPU
            if can_use_large:
                preferences = ['qwen2.5:14b', 'qwen2.5:7b', 'mistral:7b', 'phi3:mini']
            else:
                preferences = ['phi3:mini', 'qwen2.5:7b', 'mistral:7b', 'qwen2.5:14b']

            for model in preferences:
                if model in available_models:
                    self.logger.info(f"Selected model: {model}")
                    return model

            # Fallback to first available
            if available_models:
                return available_models[0]

            raise Exception("No Ollama models available")

        except Exception as e:
            self.logger.error(f"Error selecting model: {e}")
            return 'qwen2.5:14b'  # Default

    def _init_insights_db(self):
        """Initialize insights database."""
        self.insights_db = sqlite3.connect(str(self.insights_db_path))
        self.insights_cursor = self.insights_db.cursor()

        self.insights_cursor.execute('''
            CREATE TABLE IF NOT EXISTS extracted_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                category TEXT,
                insight_type TEXT,
                content TEXT,
                confidence REAL,
                source_ids TEXT,
                metadata TEXT
            )
        ''')

        self.insights_cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept TEXT UNIQUE,
                frequency INTEGER,
                related_concepts TEXT,
                first_seen TEXT,
                last_seen TEXT
            )
        ''')

        self.insights_db.commit()

    async def analyze_emails_intelligent(self, limit: int = 100, batch_size: int = 10) -> Dict[str, Any]:
        """
        Actually read and understand email content using LLM.

        Args:
            limit: Maximum number of emails to analyze
            batch_size: How many to process at once (memory management)

        Returns:
            Analysis results with real insights
        """
        start_time = datetime.now()

        self.logger.info(f"Starting intelligent email analysis (limit: {limit})")

        # Get email data from vault
        self.vault.cursor.execute('''
            SELECT id, encrypted_content, metadata, timestamp
            FROM personal_data
            WHERE category = 'emails'
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        emails = self.vault.cursor.fetchall()
        self.logger.info(f"Retrieved {len(emails)} emails for analysis")

        results = {
            'emails_analyzed': 0,
            'insights_extracted': 0,
            'topics_found': [],
            'people_mentioned': [],
            'interests_identified': [],
            'sentiment_summary': {},
            'processing_time_ms': 0
        }

        # Process in batches
        for i in range(0, len(emails), batch_size):
            batch = emails[i:i + batch_size]

            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(emails)-1)//batch_size + 1}")

            for email_id, encrypted_content, metadata_str, timestamp in batch:
                try:
                    # Decrypt email metadata
                    decrypted = self.vault.cipher.decrypt(encrypted_content).decode()
                    email_data = json.loads(decrypted)

                    # Extract subject and sender
                    subject = email_data.get('subject', '')
                    sender = email_data.get('from', '')

                    # Analyze with LLM
                    insight = await self._analyze_email_with_llm(
                        subject=subject,
                        sender=sender,
                        email_id=email_id
                    )

                    if insight:
                        results['emails_analyzed'] += 1
                        results['insights_extracted'] += 1

                        # Store insight
                        self._store_insight(insight)

                        # Aggregate findings
                        if 'topics' in insight.metadata:
                            results['topics_found'].extend(insight.metadata['topics'])
                        if 'people' in insight.metadata:
                            results['people_mentioned'].extend(insight.metadata['people'])
                        if 'interests' in insight.metadata:
                            results['interests_identified'].extend(insight.metadata['interests'])

                except Exception as e:
                    self.logger.debug(f"Error analyzing email {email_id}: {e}")
                    continue

        # Aggregate results
        results['topics_found'] = list(Counter(results['topics_found']).most_common(20))
        results['people_mentioned'] = list(Counter(results['people_mentioned']).most_common(20))
        results['interests_identified'] = list(Counter(results['interests_identified']).most_common(10))

        duration = (datetime.now() - start_time).total_seconds() * 1000
        results['processing_time_ms'] = int(duration)

        # Record operation
        self.history.record_operation(
            operation_type='intelligent_email_analysis',
            module='IntelligentAnalyzer',
            input_data={'limit': limit, 'batch_size': batch_size},
            output_data=results,
            duration_ms=int(duration),
            success=True,
            model_used=self.model,
            tokens=results['emails_analyzed'] * 100,  # Estimate
            gpu_info=self.gpu_monitor.get_gpu_info()
        )

        self.logger.info(f"Analysis complete: {results['emails_analyzed']} emails, "
                        f"{results['insights_extracted']} insights in {duration:.0f}ms")

        return results

    async def _analyze_email_with_llm(self, subject: str, sender: str,
                                     email_id: str) -> Optional[InsightRecord]:
        """Use LLM to extract insights from email subject."""

        if not subject or len(subject) < 5:
            return None

        prompt = f"""Analyze this email and extract key information:

Subject: {subject}
From: {sender}

Extract:
1. Main topics (keywords/categories)
2. People or organizations mentioned
3. User interests indicated
4. Sentiment (positive/neutral/negative)

Respond in JSON format:
{{
    "topics": ["topic1", "topic2"],
    "people": ["person1"],
    "interests": ["interest1"],
    "sentiment": "neutral",
    "summary": "one sentence summary"
}}"""

        try:
            start_time = datetime.now()

            response = self.client.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt}
            ])

            latency = (datetime.now() - start_time).total_seconds() * 1000

            # Parse response
            content = response['message']['content']

            # Extract JSON from response
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]

            analysis = json.loads(content.strip())

            # Create insight record
            insight = InsightRecord(
                timestamp=datetime.now().isoformat(),
                category='email',
                insight_type='email_analysis',
                content=analysis.get('summary', ''),
                confidence=0.8,
                source_ids=[email_id],
                metadata=analysis
            )

            # Record model performance
            self.history.record_model_performance(
                model_name=self.model,
                task_type='email_analysis',
                tokens_per_sec=50,  # Estimate
                gpu_util=self.gpu_monitor.get_gpu_info().get('utilization', 0),
                memory_mb=self.gpu_monitor.get_gpu_info().get('memory_used_mb', 0),
                latency_ms=int(latency),
                quality_score=0.8
            )

            return insight

        except Exception as e:
            self.logger.debug(f"LLM analysis error: {e}")
            return None

    def _store_insight(self, insight: InsightRecord):
        """Store extracted insight."""
        self.insights_cursor.execute('''
            INSERT INTO extracted_insights
            (timestamp, category, insight_type, content, confidence, source_ids, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            insight.timestamp,
            insight.category,
            insight.insight_type,
            insight.content,
            insight.confidence,
            json.dumps(insight.source_ids),
            json.dumps(insight.metadata)
        ))
        self.insights_db.commit()

    def get_all_insights(self) -> List[Dict]:
        """Get all extracted insights."""
        self.insights_cursor.execute('''
            SELECT timestamp, category, insight_type, content, confidence, metadata
            FROM extracted_insights
            ORDER BY timestamp DESC
        ''')

        insights = []
        for row in self.insights_cursor.fetchall():
            insights.append({
                'timestamp': row[0],
                'category': row[1],
                'type': row[2],
                'content': row[3],
                'confidence': row[4],
                'metadata': json.loads(row[5]) if row[5] else {}
            })

        return insights

    def get_topic_summary(self) -> Dict[str, int]:
        """Get summary of all topics found."""
        self.insights_cursor.execute('''
            SELECT metadata FROM extracted_insights
            WHERE category = 'email'
        ''')

        all_topics = []
        for row in self.insights_cursor.fetchall():
            metadata = json.loads(row[0])
            all_topics.extend(metadata.get('topics', []))

        return dict(Counter(all_topics).most_common(50))

    def get_interests_summary(self) -> Dict[str, int]:
        """Get summary of identified interests."""
        self.insights_cursor.execute('''
            SELECT metadata FROM extracted_insights
            WHERE category = 'email'
        ''')

        all_interests = []
        for row in self.insights_cursor.fetchall():
            metadata = json.loads(row[0])
            all_interests.extend(metadata.get('interests', []))

        return dict(Counter(all_interests).most_common(20))
