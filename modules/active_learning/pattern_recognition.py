#!/usr/bin/env python3
"""
Pattern Recognition Engine - Identifies patterns in user behavior and system usage.
Learns from repetitive actions, schedules, and workflows.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
import re
import hashlib


class PatternRecognitionEngine:
    """
    Recognizes and learns patterns in user behavior and system usage.
    """

    def __init__(self, memory_manager=None):
        """
        Initialize the Pattern Recognition Engine.

        Args:
            memory_manager: Memory manager for pattern storage
        """
        self.memory_manager = memory_manager
        self.logger = logging.getLogger("PHOENIX.PatternRecognition")

        # Pattern storage
        self.patterns = {
            'command_sequences': [],  # Sequences of commands
            'time_patterns': {},       # Time-based patterns
            'file_access': {},        # File access patterns
            'workflow_patterns': [],   # Complete workflows
            'error_patterns': [],     # Common error scenarios
            'tool_usage': {}          # Tool usage patterns
        }

        # Event history for pattern detection
        self.event_history = []
        self.command_history = []

        # Pattern detection thresholds - LOWERED for faster learning
        self.thresholds = {
            'min_occurrences': 2,      # Learn from just 2 occurrences
            'time_window': 7200,       # Wider time window (2 hours)
            'sequence_length': 10,     # Longer sequences allowed
            'confidence_threshold': 0.4  # Lower confidence to start learning
        }

        # Statistical tracking
        self.statistics = defaultdict(lambda: defaultdict(int))

    def record_event(self, event_type: str, event_data: Dict):
        """
        Record an event for pattern analysis.

        Args:
            event_type: Type of event (command, file_access, error, etc.)
            event_data: Event details
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'data': event_data,
            'id': self._generate_event_id(event_type, event_data)
        }

        self.event_history.append(event)

        # Specific handling by type
        if event_type == 'command':
            self._record_command(event_data)
        elif event_type == 'file_access':
            self._record_file_access(event_data)
        elif event_type == 'error':
            self._record_error(event_data)
        elif event_type == 'tool_use':
            self._record_tool_usage(event_data)

        # Analyze for patterns periodically
        if len(self.event_history) % 10 == 0:
            self._analyze_recent_events()

    def _record_command(self, command_data: Dict):
        """Record command execution for sequence detection."""
        self.command_history.append({
            'command': command_data.get('command'),
            'timestamp': datetime.now(),
            'directory': command_data.get('directory', ''),
            'success': command_data.get('success', True)
        })

        # Keep only recent history
        cutoff = datetime.now() - timedelta(hours=24)
        self.command_history = [
            cmd for cmd in self.command_history
            if cmd['timestamp'] > cutoff
        ]

        # Check for command sequences
        self._detect_command_sequences()

    def _record_file_access(self, file_data: Dict):
        """Record file access patterns."""
        file_path = file_data.get('path', '')
        action = file_data.get('action', 'read')

        if file_path not in self.patterns['file_access']:
            self.patterns['file_access'][file_path] = {
                'access_times': [],
                'actions': [],
                'frequency': 0
            }

        self.patterns['file_access'][file_path]['access_times'].append(datetime.now())
        self.patterns['file_access'][file_path]['actions'].append(action)
        self.patterns['file_access'][file_path]['frequency'] += 1

    def _record_error(self, error_data: Dict):
        """Record error patterns."""
        error_type = error_data.get('type', 'unknown')
        error_context = error_data.get('context', '')

        # Find similar errors
        similar_found = False
        for pattern in self.patterns['error_patterns']:
            if pattern['type'] == error_type and self._similar_context(pattern['context'], error_context):
                pattern['occurrences'] += 1
                pattern['last_seen'] = datetime.now().isoformat()
                similar_found = True
                break

        if not similar_found:
            self.patterns['error_patterns'].append({
                'type': error_type,
                'context': error_context,
                'occurrences': 1,
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat()
            })

    def _record_tool_usage(self, tool_data: Dict):
        """Record tool usage patterns."""
        tool_name = tool_data.get('tool', 'unknown')

        if tool_name not in self.patterns['tool_usage']:
            self.patterns['tool_usage'][tool_name] = {
                'usage_times': [],
                'contexts': [],
                'frequency': 0,
                'average_duration': 0
            }

        self.patterns['tool_usage'][tool_name]['usage_times'].append(datetime.now())
        self.patterns['tool_usage'][tool_name]['frequency'] += 1

        if 'context' in tool_data:
            self.patterns['tool_usage'][tool_name]['contexts'].append(tool_data['context'])

    def _detect_command_sequences(self):
        """Detect repeated command sequences."""
        if len(self.command_history) < self.thresholds['sequence_length']:
            return

        # Look for repeated sequences
        for length in range(2, min(self.thresholds['sequence_length'] + 1, len(self.command_history))):
            # Get recent commands
            recent_commands = [cmd['command'] for cmd in self.command_history[-length:]]
            sequence_hash = self._hash_sequence(recent_commands)

            # Check if we've seen this sequence before
            for pattern in self.patterns['command_sequences']:
                if pattern['hash'] == sequence_hash:
                    pattern['occurrences'] += 1
                    pattern['last_seen'] = datetime.now().isoformat()
                    pattern['confidence'] = min(1.0, pattern['occurrences'] / 10)

                    if pattern['occurrences'] >= self.thresholds['min_occurrences']:
                        self._create_workflow_suggestion(pattern)
                    break
            else:
                # New sequence
                self.patterns['command_sequences'].append({
                    'sequence': recent_commands,
                    'hash': sequence_hash,
                    'length': length,
                    'occurrences': 1,
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                    'confidence': 0.1
                })

    def _analyze_recent_events(self):
        """Analyze recent events for patterns."""
        # Time-based pattern detection
        self._detect_time_patterns()

        # Workflow detection
        self._detect_workflows()

        # Clean old events
        cutoff = datetime.now() - timedelta(hours=48)
        self.event_history = [
            event for event in self.event_history
            if datetime.fromisoformat(event['timestamp']) > cutoff
        ]

    def _detect_time_patterns(self):
        """Detect time-based patterns (daily, weekly routines)."""
        # Group events by hour of day
        hourly_events = defaultdict(list)

        for event in self.event_history:
            event_time = datetime.fromisoformat(event['timestamp'])
            hour = event_time.hour
            hourly_events[hour].append(event['type'])

        # Find peak activity hours
        for hour, events in hourly_events.items():
            if len(events) >= self.thresholds['min_occurrences']:
                event_counter = Counter(events)
                most_common = event_counter.most_common(1)[0]

                self.patterns['time_patterns'][hour] = {
                    'common_activity': most_common[0],
                    'frequency': most_common[1],
                    'confidence': most_common[1] / len(events)
                }

    def _detect_workflows(self):
        """Detect complete workflows from event sequences."""
        # Look for events that frequently occur together
        time_window = timedelta(seconds=self.thresholds['time_window'])

        # Group events by time proximity
        event_groups = []
        current_group = []

        for i, event in enumerate(self.event_history):
            if not current_group:
                current_group.append(event)
            else:
                last_time = datetime.fromisoformat(current_group[-1]['timestamp'])
                current_time = datetime.fromisoformat(event['timestamp'])

                if current_time - last_time <= time_window:
                    current_group.append(event)
                else:
                    if len(current_group) >= 2:
                        event_groups.append(current_group)
                    current_group = [event]

        # Find similar workflows
        for group in event_groups:
            workflow_signature = self._create_workflow_signature(group)

            # Check if similar workflow exists
            for pattern in self.patterns['workflow_patterns']:
                if self._similar_workflow(pattern['signature'], workflow_signature):
                    pattern['occurrences'] += 1
                    pattern['last_seen'] = datetime.now().isoformat()
                    pattern['confidence'] = min(1.0, pattern['occurrences'] / 5)
                    break
            else:
                # New workflow
                self.patterns['workflow_patterns'].append({
                    'signature': workflow_signature,
                    'events': [e['type'] for e in group],
                    'occurrences': 1,
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                    'confidence': 0.2
                })

    def predict_next_action(self, current_context: Dict) -> List[Dict[str, Any]]:
        """
        Predict likely next actions based on patterns.

        Args:
            current_context: Current system context

        Returns:
            List of predicted actions with confidence scores
        """
        predictions = []

        # Check command sequences
        if self.command_history:
            recent_commands = [cmd['command'] for cmd in self.command_history[-4:]]

            for pattern in self.patterns['command_sequences']:
                if len(pattern['sequence']) > len(recent_commands):
                    if pattern['sequence'][:len(recent_commands)] == recent_commands:
                        if pattern['confidence'] >= self.thresholds['confidence_threshold']:
                            predictions.append({
                                'type': 'command',
                                'action': pattern['sequence'][len(recent_commands)],
                                'confidence': pattern['confidence'],
                                'reason': f"Part of common sequence: {' -> '.join(pattern['sequence'])}"
                            })

        # Check time patterns
        current_hour = datetime.now().hour
        if current_hour in self.patterns['time_patterns']:
            time_pattern = self.patterns['time_patterns'][current_hour]
            if time_pattern['confidence'] >= self.thresholds['confidence_threshold']:
                predictions.append({
                    'type': 'activity',
                    'action': time_pattern['common_activity'],
                    'confidence': time_pattern['confidence'],
                    'reason': f"Common activity at {current_hour}:00"
                })

        # Check file access patterns
        for file_path, access_pattern in self.patterns['file_access'].items():
            if access_pattern['frequency'] >= self.thresholds['min_occurrences']:
                # Check if it's time to access this file again
                if access_pattern['access_times']:
                    avg_interval = self._calculate_average_interval(access_pattern['access_times'])
                    last_access = access_pattern['access_times'][-1]
                    time_since = datetime.now() - last_access

                    if time_since.total_seconds() >= avg_interval * 0.8:
                        predictions.append({
                            'type': 'file_access',
                            'action': f"Access {file_path}",
                            'confidence': min(1.0, access_pattern['frequency'] / 20),
                            'reason': f"Regular access pattern detected"
                        })

        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)

        return predictions[:5]  # Top 5 predictions

    def get_insights(self) -> Dict[str, Any]:
        """
        Get insights from detected patterns.

        Returns:
            Dictionary of insights and recommendations
        """
        insights = {
            'most_used_commands': [],
            'peak_hours': [],
            'common_workflows': [],
            'frequent_errors': [],
            'automation_opportunities': [],
            'usage_statistics': {}
        }

        # Most used commands
        if self.command_history:
            command_counter = Counter([cmd['command'].split()[0] if cmd['command'] else ''
                                      for cmd in self.command_history])
            insights['most_used_commands'] = command_counter.most_common(5)

        # Peak activity hours
        insights['peak_hours'] = [
            {'hour': hour, 'activity': pattern['common_activity']}
            for hour, pattern in self.patterns['time_patterns'].items()
            if pattern['confidence'] >= 0.5
        ]

        # Common workflows
        insights['common_workflows'] = [
            {
                'events': pattern['events'],
                'frequency': pattern['occurrences'],
                'confidence': pattern['confidence']
            }
            for pattern in self.patterns['workflow_patterns']
            if pattern['confidence'] >= self.thresholds['confidence_threshold']
        ]

        # Frequent errors
        insights['frequent_errors'] = [
            {
                'type': pattern['type'],
                'occurrences': pattern['occurrences'],
                'last_seen': pattern['last_seen']
            }
            for pattern in self.patterns['error_patterns']
            if pattern['occurrences'] >= self.thresholds['min_occurrences']
        ]

        # Automation opportunities
        for pattern in self.patterns['command_sequences']:
            if pattern['occurrences'] >= 5 and pattern['confidence'] >= 0.8:
                insights['automation_opportunities'].append({
                    'sequence': pattern['sequence'],
                    'frequency': pattern['occurrences'],
                    'suggestion': f"Create script for: {' -> '.join(pattern['sequence'])}"
                })

        return insights

    def _hash_sequence(self, sequence: List[str]) -> str:
        """Generate hash for command sequence."""
        return hashlib.md5('|'.join(sequence).encode()).hexdigest()

    def _generate_event_id(self, event_type: str, event_data: Dict) -> str:
        """Generate unique event ID."""
        data_str = f"{event_type}{json.dumps(event_data, sort_keys=True)}{datetime.now()}"
        return hashlib.md5(data_str.encode()).hexdigest()[:8]

    def _similar_context(self, context1: str, context2: str) -> bool:
        """Check if two contexts are similar."""
        # Simple similarity check - can be made more sophisticated
        return context1 == context2

    def _create_workflow_signature(self, events: List[Dict]) -> str:
        """Create signature for workflow identification."""
        event_types = [e['type'] for e in events]
        return '->'.join(event_types)

    def _similar_workflow(self, sig1: str, sig2: str) -> bool:
        """Check if two workflow signatures are similar."""
        return sig1 == sig2

    def _calculate_average_interval(self, timestamps: List[datetime]) -> float:
        """Calculate average interval between timestamps."""
        if len(timestamps) < 2:
            return 0

        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)

        return np.mean(intervals) if intervals else 0

    def _create_workflow_suggestion(self, pattern: Dict):
        """Create automation suggestion for repeated pattern."""
        self.logger.info(f"Pattern detected: {pattern['sequence']} (occurred {pattern['occurrences']} times)")

        # Store as potential automation
        if self.memory_manager:
            self.memory_manager.learn_fact(
                f"Repeated command sequence: {' -> '.join(pattern['sequence'])}",
                'automation_opportunity'
            )

    def export_patterns(self, file_path: Path):
        """Export learned patterns to file."""
        export_data = {
            'patterns': self.patterns,
            'statistics': dict(self.statistics),
            'export_time': datetime.now().isoformat()
        }

        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(f"Patterns exported to {file_path}")

    def import_patterns(self, file_path: Path):
        """Import patterns from file."""
        if not file_path.exists():
            self.logger.error(f"Pattern file not found: {file_path}")
            return

        with open(file_path, 'r') as f:
            data = json.load(f)

        self.patterns = data.get('patterns', self.patterns)
        self.statistics = defaultdict(lambda: defaultdict(int), data.get('statistics', {}))

        self.logger.info(f"Patterns imported from {file_path}")