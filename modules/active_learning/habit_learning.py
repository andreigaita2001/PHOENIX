#!/usr/bin/env python3
"""
Habit Learning System - Learns from patterns to create automations and workflows.
Converts repetitive behaviors into actionable habits and suggestions.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import subprocess
import hashlib
import re


class HabitLearner:
    """
    Learns user habits and creates automation suggestions.
    """

    def __init__(self, pattern_engine=None, memory_manager=None):
        """
        Initialize the Habit Learner.

        Args:
            pattern_engine: Pattern recognition engine
            memory_manager: Memory manager for persistence
        """
        self.pattern_engine = pattern_engine
        self.memory_manager = memory_manager
        self.logger = logging.getLogger("PHOENIX.HabitLearner")

        # Habit storage
        self.learned_habits = {
            'daily_routines': [],      # Time-based habits
            'workflow_habits': [],     # Sequence-based habits
            'context_habits': [],      # Context-dependent habits
            'tool_preferences': {},    # Preferred tools for tasks
            'automation_scripts': [],  # Generated automation scripts
            'shortcuts': {}            # Learned shortcuts and aliases
        }

        # Learning thresholds
        self.thresholds = {
            'habit_confidence': 0.5,   # Lower confidence to learn faster
            'automation_threshold': 2,  # Suggest automation after just 2 times
            'routine_consistency': 0.4,  # More flexible routine detection
            'workflow_complexity': 2     # Even 2-step workflows are valuable
        }

        # Automation templates
        self.automation_templates = self._load_automation_templates()

        # Learning state
        self.learning_enabled = True
        self.adaptation_rate = 0.1  # How quickly to adapt to changes

    def learn_from_patterns(self, patterns: Dict[str, Any]):
        """
        Learn habits from detected patterns.

        Args:
            patterns: Patterns from pattern recognition engine
        """
        if not self.learning_enabled:
            return

        # Learn from command sequences
        if 'command_sequences' in patterns:
            self._learn_workflow_habits(patterns['command_sequences'])

        # Learn from time patterns
        if 'time_patterns' in patterns:
            self._learn_daily_routines(patterns['time_patterns'])

        # Learn from file access patterns
        if 'file_access' in patterns:
            self._learn_file_habits(patterns['file_access'])

        # Learn from tool usage
        if 'tool_usage' in patterns:
            self._learn_tool_preferences(patterns['tool_usage'])

        # Learn from error patterns
        if 'error_patterns' in patterns:
            self._learn_error_avoidance(patterns['error_patterns'])

    def _learn_workflow_habits(self, command_sequences: List[Dict]):
        """Learn habits from command sequences."""
        for sequence in command_sequences:
            if sequence['confidence'] >= self.thresholds['habit_confidence']:
                if sequence['occurrences'] >= self.thresholds['automation_threshold']:
                    # Create workflow habit
                    habit = {
                        'type': 'workflow',
                        'name': self._generate_habit_name(sequence['sequence']),
                        'commands': sequence['sequence'],
                        'frequency': sequence['occurrences'],
                        'confidence': sequence['confidence'],
                        'last_seen': sequence['last_seen'],
                        'automation_ready': True
                    }

                    # Check if habit already exists
                    if not self._habit_exists(habit):
                        self.learned_habits['workflow_habits'].append(habit)
                        self.logger.info(f"Learned workflow habit: {habit['name']}")

                        # Generate automation script
                        self._create_automation_script(habit)

    def _learn_daily_routines(self, time_patterns: Dict[int, Dict]):
        """Learn daily routine habits."""
        for hour, pattern in time_patterns.items():
            if pattern['confidence'] >= self.thresholds['routine_consistency']:
                routine = {
                    'type': 'daily_routine',
                    'hour': hour,
                    'activity': pattern['common_activity'],
                    'frequency': pattern['frequency'],
                    'confidence': pattern['confidence'],
                    'suggestion': f"At {hour}:00, you typically {pattern['common_activity']}"
                }

                # Add or update routine
                existing = next((r for r in self.learned_habits['daily_routines']
                               if r['hour'] == hour), None)

                if existing:
                    # Update confidence with moving average
                    existing['confidence'] = (existing['confidence'] * (1 - self.adaptation_rate) +
                                             pattern['confidence'] * self.adaptation_rate)
                else:
                    self.learned_habits['daily_routines'].append(routine)
                    self.logger.info(f"Learned daily routine: {routine['suggestion']}")

    def _learn_file_habits(self, file_patterns: Dict[str, Dict]):
        """Learn file access habits."""
        for file_path, access_pattern in file_patterns.items():
            if access_pattern['frequency'] >= self.thresholds['automation_threshold']:
                # Determine file habit type
                actions = access_pattern.get('actions', [])

                if 'edit' in actions and 'read' in actions:
                    habit_type = 'frequent_edit'
                elif 'read' in actions:
                    habit_type = 'frequent_read'
                else:
                    habit_type = 'frequent_access'

                habit = {
                    'type': 'file_habit',
                    'file': file_path,
                    'habit_type': habit_type,
                    'frequency': access_pattern['frequency'],
                    'last_access': access_pattern['access_times'][-1].isoformat()
                                  if access_pattern['access_times'] else None,
                    'suggestion': self._generate_file_suggestion(file_path, habit_type)
                }

                # Store as context habit
                self.learned_habits['context_habits'].append(habit)

    def _learn_tool_preferences(self, tool_usage: Dict[str, Dict]):
        """Learn tool preferences from usage patterns."""
        for tool_name, usage_data in tool_usage.items():
            if usage_data['frequency'] >= 3:  # Minimum usage to establish preference
                # Analyze contexts to determine when tool is used
                contexts = usage_data.get('contexts', [])

                if contexts:
                    # Find common patterns in contexts
                    common_keywords = self._extract_common_keywords(contexts)

                    self.learned_habits['tool_preferences'][tool_name] = {
                        'frequency': usage_data['frequency'],
                        'common_contexts': common_keywords,
                        'average_duration': usage_data.get('average_duration', 0),
                        'suggestion': f"Use {tool_name} for: {', '.join(common_keywords[:3])}"
                    }

    def _learn_error_avoidance(self, error_patterns: List[Dict]):
        """Learn to avoid common errors."""
        for error in error_patterns:
            if error['occurrences'] >= 3:
                # Create error avoidance habit
                avoidance = {
                    'type': 'error_avoidance',
                    'error_type': error['type'],
                    'context': error.get('context', ''),
                    'occurrences': error['occurrences'],
                    'prevention': self._suggest_error_prevention(error),
                    'last_seen': error['last_seen']
                }

                self.learned_habits['context_habits'].append(avoidance)
                self.logger.info(f"Learned error avoidance: {error['type']}")

    def generate_automations(self) -> List[Dict[str, Any]]:
        """
        Generate automation suggestions from learned habits.

        Returns:
            List of automation suggestions
        """
        automations = []

        # Generate from workflow habits
        for habit in self.learned_habits['workflow_habits']:
            if habit.get('automation_ready'):
                automation = {
                    'name': habit['name'],
                    'type': 'shell_script',
                    'commands': habit['commands'],
                    'frequency': habit['frequency'],
                    'script': self._create_shell_script(habit['commands']),
                    'alias_suggestion': self._suggest_alias(habit['name'])
                }
                automations.append(automation)

        # Generate from daily routines
        for routine in self.learned_habits['daily_routines']:
            if routine['confidence'] >= self.thresholds['routine_consistency']:
                automation = {
                    'name': f"daily_{routine['activity']}_{routine['hour']}h",
                    'type': 'scheduled_task',
                    'schedule': f"0 {routine['hour']} * * *",  # Cron format
                    'activity': routine['activity'],
                    'cron_entry': self._create_cron_entry(routine)
                }
                automations.append(automation)

        # Generate shortcuts from tool preferences
        for tool, prefs in self.learned_habits['tool_preferences'].items():
            if prefs['frequency'] >= 10:
                automation = {
                    'name': f"quick_{tool}",
                    'type': 'tool_shortcut',
                    'tool': tool,
                    'contexts': prefs['common_contexts'],
                    'function': self._create_tool_function(tool, prefs)
                }
                automations.append(automation)

        return automations

    def _create_automation_script(self, habit: Dict):
        """Create an automation script from a habit."""
        script_name = habit['name'].replace(' ', '_').lower()
        script_content = self._create_shell_script(habit['commands'])

        script_entry = {
            'name': script_name,
            'content': script_content,
            'created': datetime.now().isoformat(),
            'frequency': habit['frequency'],
            'commands': habit['commands']
        }

        self.learned_habits['automation_scripts'].append(script_entry)

        # Save to file if configured
        if self.memory_manager:
            script_path = Path("/tmp") / f"phoenix_automation_{script_name}.sh"
            with open(script_path, 'w') as f:
                f.write(script_content)
            script_path.chmod(0o755)

            self.logger.info(f"Created automation script: {script_path}")

    def _create_shell_script(self, commands: List[str]) -> str:
        """Create a shell script from commands."""
        script = """#!/bin/bash
# PHOENIX Generated Automation Script
# Created: {date}
# Based on learned habit

set -e  # Exit on error

""".format(date=datetime.now().isoformat())

        for cmd in commands:
            # Add error handling for each command
            script += f"""
echo "Executing: {cmd}"
{cmd}
if [ $? -ne 0 ]; then
    echo "Error executing: {cmd}"
    exit 1
fi
"""

        script += """
echo "Automation completed successfully!"
"""
        return script

    def _create_cron_entry(self, routine: Dict) -> str:
        """Create a cron entry for a routine."""
        return f"0 {routine['hour']} * * * /usr/bin/phoenix_routine {routine['activity']}"

    def _create_tool_function(self, tool: str, preferences: Dict) -> str:
        """Create a function wrapper for tool usage."""
        return f"""
function quick_{tool}() {{
    # Quick access to {tool} based on learned preferences
    # Common contexts: {', '.join(preferences['common_contexts'][:3])}

    phoenix_tool {tool} "$@"
}}
"""

    def suggest_next_action(self, current_context: Dict) -> List[Dict[str, Any]]:
        """
        Suggest next actions based on learned habits.

        Args:
            current_context: Current system context

        Returns:
            List of suggestions with confidence scores
        """
        suggestions = []

        # Check daily routines
        current_hour = datetime.now().hour
        for routine in self.learned_habits['daily_routines']:
            if routine['hour'] == current_hour:
                suggestions.append({
                    'type': 'routine',
                    'action': routine['activity'],
                    'confidence': routine['confidence'],
                    'reason': routine['suggestion']
                })

        # Check workflow applicability
        recent_commands = current_context.get('recent_commands', [])
        for habit in self.learned_habits['workflow_habits']:
            if self._workflow_matches_context(habit, recent_commands):
                suggestions.append({
                    'type': 'workflow',
                    'action': f"Run workflow: {habit['name']}",
                    'commands': habit['commands'],
                    'confidence': habit['confidence'],
                    'reason': f"Frequently used sequence ({habit['frequency']} times)"
                })

        # Check file habits
        current_directory = current_context.get('directory', '')
        for habit in self.learned_habits['context_habits']:
            if habit.get('type') == 'file_habit':
                if current_directory and habit['file'].startswith(current_directory):
                    suggestions.append({
                        'type': 'file_access',
                        'action': habit['suggestion'],
                        'file': habit['file'],
                        'confidence': 0.7,
                        'reason': f"Frequently accessed ({habit['frequency']} times)"
                    })

        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)

        return suggestions[:3]  # Top 3 suggestions

    def adapt_habit(self, habit_id: str, feedback: str):
        """
        Adapt a habit based on user feedback.

        Args:
            habit_id: Identifier for the habit
            feedback: User feedback (positive/negative/modify)
        """
        # Find and update habit based on feedback
        if feedback == 'positive':
            # Increase confidence
            self._boost_habit_confidence(habit_id)
        elif feedback == 'negative':
            # Decrease confidence or remove
            self._reduce_habit_confidence(habit_id)
        elif feedback.startswith('modify:'):
            # Modify the habit
            modification = feedback.replace('modify:', '').strip()
            self._modify_habit(habit_id, modification)

        self.logger.info(f"Adapted habit {habit_id} based on {feedback} feedback")

    def _generate_habit_name(self, commands: List[str]) -> str:
        """Generate a descriptive name for a habit."""
        if not commands:
            return "unknown_habit"

        # Extract key commands
        key_words = []
        for cmd in commands[:3]:  # Look at first 3 commands
            parts = cmd.split()
            if parts:
                key_words.append(parts[0])

        return f"workflow_{'_'.join(key_words)}"

    def _habit_exists(self, new_habit: Dict) -> bool:
        """Check if a habit already exists."""
        for habit in self.learned_habits['workflow_habits']:
            if habit['commands'] == new_habit['commands']:
                return True
        return False

    def _generate_file_suggestion(self, file_path: str, habit_type: str) -> str:
        """Generate suggestion for file habit."""
        file_name = Path(file_path).name

        if habit_type == 'frequent_edit':
            return f"Open {file_name} for editing"
        elif habit_type == 'frequent_read':
            return f"Review {file_name}"
        else:
            return f"Access {file_name}"

    def _extract_common_keywords(self, contexts: List[str]) -> List[str]:
        """Extract common keywords from contexts."""
        word_count = {}

        for context in contexts:
            words = re.findall(r'\b[a-z]+\b', context.lower())
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_count[word] = word_count.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:5]]

    def _suggest_error_prevention(self, error: Dict) -> str:
        """Suggest how to prevent an error."""
        error_type = error['type']

        if 'permission' in error_type.lower():
            return "Check file permissions before operation"
        elif 'not found' in error_type.lower():
            return "Verify file/command exists before use"
        elif 'syntax' in error_type.lower():
            return "Validate syntax before execution"
        else:
            return f"Be cautious with {error_type} scenarios"

    def _suggest_alias(self, name: str) -> str:
        """Suggest an alias for a workflow."""
        # Create short alias from name
        parts = name.split('_')
        if len(parts) >= 2:
            # Use first letters
            alias = ''.join(p[0] for p in parts if p)
        else:
            alias = name[:3]

        return f"alias {alias}='phoenix_run {name}'"

    def _workflow_matches_context(self, habit: Dict, recent_commands: List[str]) -> bool:
        """Check if a workflow matches current context."""
        if not recent_commands:
            return False

        # Check if recent commands match beginning of workflow
        workflow_start = habit['commands'][:2]
        recent_relevant = recent_commands[-2:] if len(recent_commands) >= 2 else recent_commands

        return workflow_start[:len(recent_relevant)] == recent_relevant

    def _boost_habit_confidence(self, habit_id: str):
        """Increase confidence in a habit."""
        # Search all habit categories
        for category in self.learned_habits.values():
            if isinstance(category, list):
                for habit in category:
                    if habit.get('name') == habit_id:
                        habit['confidence'] = min(1.0, habit['confidence'] + 0.1)
                        return

    def _reduce_habit_confidence(self, habit_id: str):
        """Reduce confidence in a habit."""
        for category in self.learned_habits.values():
            if isinstance(category, list):
                for habit in category:
                    if habit.get('name') == habit_id:
                        habit['confidence'] *= 0.8
                        if habit['confidence'] < 0.3:
                            category.remove(habit)
                        return

    def _modify_habit(self, habit_id: str, modification: str):
        """Modify a habit based on user input."""
        # This would parse the modification and update the habit
        # For now, log the modification request
        self.logger.info(f"Modification requested for {habit_id}: {modification}")

    def _load_automation_templates(self) -> Dict[str, str]:
        """Load automation templates."""
        return {
            'backup': """#!/bin/bash
# Automated backup script
tar -czf backup_$(date +%Y%m%d).tar.gz {directories}
""",
            'cleanup': """#!/bin/bash
# Automated cleanup script
find {directory} -type f -mtime +{days} -delete
""",
            'monitor': """#!/bin/bash
# System monitoring script
while true; do
    {commands}
    sleep {interval}
done
"""
        }

    def export_habits(self, file_path: Path):
        """Export learned habits to file."""
        export_data = {
            'habits': self.learned_habits,
            'thresholds': self.thresholds,
            'export_time': datetime.now().isoformat()
        }

        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(f"Habits exported to {file_path}")

    def import_habits(self, file_path: Path):
        """Import habits from file."""
        if not file_path.exists():
            self.logger.error(f"Habit file not found: {file_path}")
            return

        with open(file_path, 'r') as f:
            data = json.load(f)

        self.learned_habits = data.get('habits', self.learned_habits)
        self.logger.info(f"Habits imported from {file_path}")

    def get_habit_summary(self) -> Dict[str, Any]:
        """Get summary of learned habits."""
        return {
            'total_workflows': len(self.learned_habits['workflow_habits']),
            'daily_routines': len(self.learned_habits['daily_routines']),
            'context_habits': len(self.learned_habits['context_habits']),
            'tool_preferences': len(self.learned_habits['tool_preferences']),
            'automation_scripts': len(self.learned_habits['automation_scripts']),
            'shortcuts': len(self.learned_habits['shortcuts']),
            'learning_enabled': self.learning_enabled,
            'adaptation_rate': self.adaptation_rate
        }