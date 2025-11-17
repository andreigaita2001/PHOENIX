#!/usr/bin/env python3
"""
Advanced Learning Module - Makes Phoenix truly adaptive and intelligent.
This module enables autonomous learning from interactions, patterns, and feedback.
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging
import hashlib
import re


class AdvancedLearning:
    """
    Enables Phoenix to learn autonomously from every interaction.
    Adapts behavior, predicts needs, and improves continuously.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Advanced Learning system.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("PHOENIX.Learning")

        # Learning data directory
        self.data_dir = Path(config.get('learning_dir', './data/learning'))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize learning components
        self.pattern_memory = self._load_patterns()
        self.command_history = self._load_history()
        self.user_model = self._load_user_model()
        self.skill_confidence = self._load_skills()
        self.feedback_history = []

        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.1)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.pattern_threshold = 3  # Min occurrences to learn pattern

        # Autonomous learning settings
        self.auto_learn = config.get('auto_learn', True)
        self.auto_optimize = config.get('auto_optimize', True)
        self.auto_suggest = config.get('auto_suggest', True)

        self.logger.info("Advanced Learning system initialized")

    # ============= PATTERN LEARNING =============

    def learn_from_interaction(self, command: str, response: str, success: bool = True):
        """
        Learn from each interaction autonomously.

        Args:
            command: User's command
            response: Phoenix's response
            success: Whether the interaction was successful
        """
        # Record the interaction
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'command': command,
            'response': response,
            'success': success,
            'context': self._get_current_context()
        }

        self.command_history.append(interaction)

        # Learn patterns
        self._learn_command_patterns(command, success)
        self._learn_temporal_patterns(command)
        self._learn_sequence_patterns()

        # Update user model
        self._update_user_model(command, response, success)

        # Improve skills
        if success:
            self._improve_skill_confidence(command)
        else:
            self._reduce_skill_confidence(command)

        # Auto-optimize if enabled
        if self.auto_optimize:
            self._optimize_responses()

        # Save learnings
        if len(self.command_history) % 10 == 0:  # Save every 10 interactions
            self.save_learnings()

    def _learn_command_patterns(self, command: str, success: bool):
        """
        Learn patterns in user commands.

        Args:
            command: The command to learn from
            success: Whether it was successful
        """
        # Extract key phrases
        words = command.lower().split()

        # Learn word associations
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            self.pattern_memory['bigrams'][bigram] += 1

        # Learn command types
        if 'check' in command.lower() or 'show' in command.lower():
            self.pattern_memory['command_types']['query'] += 1
        elif 'run' in command.lower() or 'execute' in command.lower():
            self.pattern_memory['command_types']['action'] += 1
        elif 'help' in command.lower() or 'what' in command.lower():
            self.pattern_memory['command_types']['help'] += 1

        # Learn success patterns
        if success:
            self.pattern_memory['successful_patterns'].append(command)
        else:
            self.pattern_memory['failed_patterns'].append(command)

    def _learn_temporal_patterns(self, command: str):
        """
        Learn when certain commands are typically used.

        Args:
            command: The command to analyze
        """
        current_time = datetime.now()
        hour = current_time.hour
        day = current_time.weekday()

        # Learn hourly patterns
        self.pattern_memory['temporal']['hourly'][hour].append(command)

        # Learn daily patterns
        self.pattern_memory['temporal']['daily'][day].append(command)

        # Learn command frequency
        self.pattern_memory['temporal']['frequency'][command] = \
            self.pattern_memory['temporal']['frequency'].get(command, 0) + 1

    def _learn_sequence_patterns(self):
        """
        Learn sequences of commands that often occur together.
        """
        if len(self.command_history) < 2:
            return

        # Look at last 10 commands
        recent = self.command_history[-10:]

        for i in range(len(recent) - 1):
            cmd1 = recent[i]['command']
            cmd2 = recent[i + 1]['command']

            # Store sequence
            sequence_key = f"{self._normalize_command(cmd1)} -> {self._normalize_command(cmd2)}"
            self.pattern_memory['sequences'][sequence_key] += 1

    # ============= PREDICTION & SUGGESTION =============

    def predict_next_command(self, current_command: str = None) -> List[Tuple[str, float]]:
        """
        Predict what the user might want to do next.

        Args:
            current_command: The current command (if any)

        Returns:
            List of (predicted_command, confidence) tuples
        """
        predictions = []

        # Time-based predictions
        current_hour = datetime.now().hour
        hourly_patterns = self.pattern_memory['temporal']['hourly'].get(current_hour, [])
        if hourly_patterns:
            # Get most common commands at this hour
            hour_counter = Counter(hourly_patterns)
            for cmd, count in hour_counter.most_common(3):
                confidence = count / len(hourly_patterns)
                predictions.append((cmd, confidence * 0.5))  # Weight temporal predictions lower

        # Sequence-based predictions
        if current_command:
            normalized = self._normalize_command(current_command)
            for sequence, count in self.pattern_memory['sequences'].items():
                if sequence.startswith(normalized + " ->"):
                    next_cmd = sequence.split(" -> ")[1]
                    confidence = count / self.pattern_threshold
                    predictions.append((next_cmd, min(confidence, 0.9)))

        # User model predictions
        if self.user_model.get('preferred_commands'):
            for cmd, frequency in self.user_model['preferred_commands'].most_common(3):
                confidence = frequency / sum(self.user_model['preferred_commands'].values())
                predictions.append((cmd, confidence * 0.7))

        # Sort by confidence and remove duplicates
        seen = set()
        unique_predictions = []
        for cmd, conf in sorted(predictions, key=lambda x: x[1], reverse=True):
            if cmd not in seen:
                seen.add(cmd)
                unique_predictions.append((cmd, conf))

        return unique_predictions[:5]  # Top 5 predictions

    def suggest_improvement(self, command: str) -> Optional[str]:
        """
        Suggest a better way to phrase or execute a command.

        Args:
            command: The user's command

        Returns:
            Suggested improvement or None
        """
        # Check if we've seen similar but more successful commands
        command_lower = command.lower()
        best_match = None
        best_score = 0

        for successful_cmd in self.pattern_memory['successful_patterns']:
            similarity = self._calculate_similarity(command_lower, successful_cmd.lower())
            if similarity > 0.7 and similarity > best_score:
                best_match = successful_cmd
                best_score = similarity

        if best_match and best_match != command:
            return f"💡 Tip: You might want to try: '{best_match}' (worked well before)"

        return None

    # ============= USER MODELING =============

    def _update_user_model(self, command: str, response: str, success: bool):
        """
        Update the model of user preferences and behavior.

        Args:
            command: User's command
            response: Phoenix's response
            success: Whether interaction was successful
        """
        # Track command preferences
        normalized = self._normalize_command(command)
        self.user_model['preferred_commands'][normalized] += 1

        # Track interaction style
        if len(command.split()) < 5:
            self.user_model['style']['brief'] += 1
        else:
            self.user_model['style']['detailed'] += 1

        # Track success rate
        self.user_model['success_rate'] = (
            self.user_model.get('success_rate', 0) * 0.9 + (1.0 if success else 0.0) * 0.1
        )

        # Track expertise level
        if success and 'complex' in self._classify_command_complexity(command):
            self.user_model['expertise_level'] = min(
                self.user_model.get('expertise_level', 0.5) + 0.01, 1.0
            )

        # Track interests
        topics = self._extract_topics(command)
        for topic in topics:
            self.user_model['interests'][topic] += 1

    def get_user_expertise_level(self) -> float:
        """
        Get the estimated user expertise level (0-1).

        Returns:
            Expertise level
        """
        return self.user_model.get('expertise_level', 0.5)

    def get_user_preferences(self) -> Dict[str, Any]:
        """
        Get learned user preferences.

        Returns:
            Dictionary of preferences
        """
        return {
            'communication_style': 'brief' if self.user_model['style']['brief'] > self.user_model['style']['detailed'] else 'detailed',
            'favorite_commands': self.user_model['preferred_commands'].most_common(5),
            'interests': self.user_model['interests'].most_common(5),
            'expertise_level': self.get_user_expertise_level(),
            'success_rate': self.user_model.get('success_rate', 0)
        }

    # ============= SKILL IMPROVEMENT =============

    def _improve_skill_confidence(self, command: str):
        """
        Increase confidence in skills used successfully.

        Args:
            command: Successful command
        """
        skill = self._identify_skill(command)
        if skill:
            current = self.skill_confidence.get(skill, 0.5)
            self.skill_confidence[skill] = min(current + self.learning_rate, 1.0)
            self.logger.debug(f"Improved {skill} confidence to {self.skill_confidence[skill]:.2f}")

    def _reduce_skill_confidence(self, command: str):
        """
        Reduce confidence in skills that failed.

        Args:
            command: Failed command
        """
        skill = self._identify_skill(command)
        if skill:
            current = self.skill_confidence.get(skill, 0.5)
            self.skill_confidence[skill] = max(current - self.learning_rate / 2, 0.1)
            self.logger.debug(f"Reduced {skill} confidence to {self.skill_confidence[skill]:.2f}")

    def get_skill_confidence(self, skill: str) -> float:
        """
        Get confidence level for a specific skill.

        Args:
            skill: Skill name

        Returns:
            Confidence level (0-1)
        """
        return self.skill_confidence.get(skill, 0.5)

    # ============= AUTONOMOUS OPTIMIZATION =============

    def _optimize_responses(self):
        """
        Autonomously optimize response strategies based on success patterns.
        """
        if len(self.command_history) < 50:
            return  # Need more data

        # Analyze recent performance
        recent = self.command_history[-50:]
        success_rate = sum(1 for r in recent if r['success']) / len(recent)

        if success_rate < 0.7:
            # Performance is low, analyze failures
            failures = [r for r in recent if not r['success']]
            failure_patterns = Counter([self._normalize_command(f['command']) for f in failures])

            # Store patterns to avoid
            self.pattern_memory['avoid_patterns'] = failure_patterns.most_common(5)
            self.logger.info(f"Identified {len(failure_patterns)} failure patterns to avoid")

    def generate_learning_report(self) -> str:
        """
        Generate a report of what Phoenix has learned.

        Returns:
            Learning report string
        """
        report = "📊 **Learning Report**\n\n"

        # Command statistics
        total_commands = len(self.command_history)
        if total_commands > 0:
            success_rate = sum(1 for c in self.command_history if c['success']) / total_commands
            report += f"**Total Interactions:** {total_commands}\n"
            report += f"**Success Rate:** {success_rate:.1%}\n\n"

        # User preferences
        prefs = self.get_user_preferences()
        report += f"**Your Style:** {prefs['communication_style']}\n"
        report += f"**Expertise Level:** {prefs['expertise_level']:.1%}\n\n"

        # Favorite commands
        if prefs['favorite_commands']:
            report += "**Your Top Commands:**\n"
            for cmd, count in prefs['favorite_commands']:
                report += f"  • {cmd} ({count} times)\n"
            report += "\n"

        # Learned patterns
        if self.pattern_memory['sequences']:
            report += "**Learned Sequences:**\n"
            for seq, count in sorted(self.pattern_memory['sequences'].items(), key=lambda x: x[1], reverse=True)[:3]:
                if count >= self.pattern_threshold:
                    report += f"  • {seq} ({count} times)\n"
            report += "\n"

        # Skills
        if self.skill_confidence:
            report += "**Skill Confidence:**\n"
            for skill, conf in sorted(self.skill_confidence.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
                report += f"  • {skill}: {bar} {conf:.0%}\n"

        return report

    # ============= HELPER METHODS =============

    def _normalize_command(self, command: str) -> str:
        """Normalize command for pattern matching."""
        # Remove specific values but keep structure
        normalized = re.sub(r'\b\d+\b', 'NUM', command.lower())
        normalized = re.sub(r'/[\w/]+', 'PATH', normalized)
        normalized = re.sub(r'\b\w+\.\w+\b', 'FILE', normalized)
        return normalized.strip()

    def _calculate_similarity(self, cmd1: str, cmd2: str) -> float:
        """Calculate similarity between two commands."""
        words1 = set(cmd1.split())
        words2 = set(cmd2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)

    def _identify_skill(self, command: str) -> Optional[str]:
        """Identify which skill a command uses."""
        command_lower = command.lower()
        if any(word in command_lower for word in ['gpu', 'cpu', 'memory', 'disk']):
            return 'system_monitoring'
        elif any(word in command_lower for word in ['list', 'show', 'file']):
            return 'file_management'
        elif any(word in command_lower for word in ['run', 'execute', 'command']):
            return 'command_execution'
        elif any(word in command_lower for word in ['remember', 'learn', 'know']):
            return 'memory_management'
        return 'general'

    def _classify_command_complexity(self, command: str) -> str:
        """Classify command complexity."""
        words = command.split()
        if len(words) > 10 or any(op in command for op in ['&&', '||', '|', '>']):
            return 'complex'
        elif len(words) > 5:
            return 'moderate'
        return 'simple'

    def _extract_topics(self, command: str) -> List[str]:
        """Extract topics from command."""
        topics = []
        topic_keywords = {
            'programming': ['code', 'python', 'script', 'function', 'class'],
            'system': ['cpu', 'gpu', 'memory', 'disk', 'process'],
            'files': ['file', 'directory', 'folder', 'path'],
            'learning': ['remember', 'learn', 'know', 'teach'],
            'automation': ['schedule', 'automate', 'monitor', 'watch']
        }

        command_lower = command.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in command_lower for kw in keywords):
                topics.append(topic)

        return topics

    def _get_current_context(self) -> Dict[str, Any]:
        """Get current context for learning."""
        return {
            'time': datetime.now().isoformat(),
            'hour': datetime.now().hour,
            'day': datetime.now().weekday(),
            'recent_commands': len(self.command_history)
        }

    # ============= PERSISTENCE =============

    def save_learnings(self):
        """Save all learned patterns and models."""
        # Save patterns
        with open(self.data_dir / 'patterns.json', 'w') as f:
            # Convert defaultdicts to regular dicts for JSON serialization
            patterns_to_save = {
                'bigrams': dict(self.pattern_memory['bigrams']),
                'command_types': dict(self.pattern_memory['command_types']),
                'successful_patterns': self.pattern_memory['successful_patterns'][-100:],  # Keep last 100
                'failed_patterns': self.pattern_memory['failed_patterns'][-50:],  # Keep last 50
                'sequences': dict(self.pattern_memory['sequences']),
                'temporal': {
                    'hourly': {str(k): v for k, v in self.pattern_memory['temporal']['hourly'].items()},
                    'daily': {str(k): v for k, v in self.pattern_memory['temporal']['daily'].items()},
                    'frequency': dict(self.pattern_memory['temporal']['frequency'])
                }
            }
            json.dump(patterns_to_save, f, indent=2)

        # Save command history (keep last 1000)
        with open(self.data_dir / 'history.json', 'w') as f:
            json.dump(self.command_history[-1000:], f, indent=2)

        # Save user model
        with open(self.data_dir / 'user_model.pkl', 'wb') as f:
            pickle.dump(self.user_model, f)

        # Save skills
        with open(self.data_dir / 'skills.json', 'w') as f:
            json.dump(self.skill_confidence, f, indent=2)

        self.logger.info("Saved learning data")

    def _load_patterns(self) -> Dict:
        """Load saved patterns."""
        patterns_file = self.data_dir / 'patterns.json'
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                data = json.load(f)
                # Convert back to defaultdicts
                return {
                    'bigrams': defaultdict(int, data.get('bigrams', {})),
                    'command_types': defaultdict(int, data.get('command_types', {})),
                    'successful_patterns': data.get('successful_patterns', []),
                    'failed_patterns': data.get('failed_patterns', []),
                    'sequences': defaultdict(int, data.get('sequences', {})),
                    'temporal': {
                        'hourly': defaultdict(list, {int(k): v for k, v in data.get('temporal', {}).get('hourly', {}).items()}),
                        'daily': defaultdict(list, {int(k): v for k, v in data.get('temporal', {}).get('daily', {}).items()}),
                        'frequency': defaultdict(int, data.get('temporal', {}).get('frequency', {}))
                    },
                    'avoid_patterns': []
                }

        return {
            'bigrams': defaultdict(int),
            'command_types': defaultdict(int),
            'successful_patterns': [],
            'failed_patterns': [],
            'sequences': defaultdict(int),
            'temporal': {
                'hourly': defaultdict(list),
                'daily': defaultdict(list),
                'frequency': defaultdict(int)
            },
            'avoid_patterns': []
        }

    def _load_history(self) -> List[Dict]:
        """Load command history."""
        history_file = self.data_dir / 'history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                return json.load(f)
        return []

    def _load_user_model(self) -> Dict:
        """Load user model."""
        model_file = self.data_dir / 'user_model.pkl'
        if model_file.exists():
            with open(model_file, 'rb') as f:
                return pickle.load(f)

        return {
            'preferred_commands': Counter(),
            'style': {'brief': 0, 'detailed': 0},
            'interests': Counter(),
            'expertise_level': 0.5,
            'success_rate': 0.5
        }

    def _load_skills(self) -> Dict[str, float]:
        """Load skill confidence levels."""
        skills_file = self.data_dir / 'skills.json'
        if skills_file.exists():
            with open(skills_file, 'r') as f:
                return json.load(f)
        return {}