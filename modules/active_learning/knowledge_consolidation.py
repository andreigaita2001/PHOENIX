#!/usr/bin/env python3
"""
Knowledge Consolidation - Organizes and optimizes learned knowledge.
Converts patterns, habits, and predictions into actionable intelligence.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from collections import defaultdict
import sqlite3
import pickle
import hashlib


class KnowledgeConsolidator:
    """
    Consolidates learned knowledge into organized, actionable intelligence.
    """

    def __init__(self, scanner=None, pattern_engine=None, habit_learner=None,
                 predictive_model=None, memory_manager=None):
        """
        Initialize the Knowledge Consolidator.

        Args:
            scanner: System scanner
            pattern_engine: Pattern recognition engine
            habit_learner: Habit learning system
            predictive_model: Predictive modeling system
            memory_manager: Memory manager for persistence
        """
        self.scanner = scanner
        self.pattern_engine = pattern_engine
        self.habit_learner = habit_learner
        self.predictive_model = predictive_model
        self.memory_manager = memory_manager
        self.logger = logging.getLogger("PHOENIX.KnowledgeConsolidator")

        # Knowledge base
        self.knowledge_base = {
            'system_facts': {},       # Static system information
            'behavioral_patterns': {},  # User behavior patterns
            'workflows': {},          # Learned workflows
            'optimizations': {},      # System optimizations
            'relationships': {},      # Entity relationships
            'insights': []           # High-level insights
        }

        # Knowledge graph
        self.knowledge_graph = KnowledgeGraph()

        # Consolidation rules
        self.consolidation_rules = self._load_consolidation_rules()

        # Knowledge quality metrics
        self.quality_metrics = {
            'accuracy': 0.0,
            'completeness': 0.0,
            'consistency': 0.0,
            'relevance': 0.0,
            'timeliness': 0.0
        }

        # Initialize knowledge database
        self.db_path = Path("/tmp/phoenix_knowledge.db")
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for knowledge storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id TEXT PRIMARY KEY,
                category TEXT,
                type TEXT,
                content TEXT,
                confidence REAL,
                source TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                entity1 TEXT,
                entity2 TEXT,
                relationship_type TEXT,
                strength REAL,
                created_at TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS insights (
                id TEXT PRIMARY KEY,
                insight TEXT,
                category TEXT,
                importance REAL,
                actionable BOOLEAN,
                created_at TIMESTAMP,
                acted_upon BOOLEAN DEFAULT 0
            )
        ''')

        conn.commit()
        conn.close()

    def consolidate_all_knowledge(self) -> Dict[str, Any]:
        """
        Consolidate knowledge from all learning systems.

        Returns:
            Consolidation summary
        """
        self.logger.info("Starting knowledge consolidation...")

        # Gather knowledge from all sources
        system_knowledge = self._consolidate_system_knowledge()
        pattern_knowledge = self._consolidate_pattern_knowledge()
        habit_knowledge = self._consolidate_habit_knowledge()
        predictive_knowledge = self._consolidate_predictive_knowledge()

        # Merge and organize
        self._merge_knowledge(
            system_knowledge,
            pattern_knowledge,
            habit_knowledge,
            predictive_knowledge
        )

        # Extract relationships
        self._extract_relationships()

        # Generate insights
        insights = self._generate_insights()

        # Optimize knowledge base
        self._optimize_knowledge_base()

        # Calculate quality metrics
        self._calculate_quality_metrics()

        # Store in database
        self._persist_knowledge()

        summary = {
            'total_items': self._count_knowledge_items(),
            'relationships': len(self.knowledge_graph.edges),
            'insights': len(insights),
            'quality_metrics': self.quality_metrics,
            'timestamp': datetime.now().isoformat()
        }

        self.logger.info(f"Consolidation complete: {summary['total_items']} items")
        return summary

    def _consolidate_system_knowledge(self) -> Dict[str, Any]:
        """Consolidate system scanner knowledge."""
        if not self.scanner:
            return {}

        # Get system map from scanner
        system_info = self.scanner.system_map
        if not system_info:
            return {}

        return {
            'system': {
                'os': system_info.get('os_info', {}),
                'hardware': system_info.get('hardware', {}),
                'software': system_info.get('installed_software', []),
                'services': system_info.get('services', []),
                'network': system_info.get('network', {})
            },
            'projects': system_info.get('projects', []),
            'tools': system_info.get('development_tools', [])
        }

    def _consolidate_pattern_knowledge(self) -> Dict[str, Any]:
        """Consolidate pattern recognition knowledge."""
        if not self.pattern_engine:
            return {}

        patterns = self.pattern_engine.patterns

        # Filter and organize significant patterns
        significant_patterns = {
            'command_sequences': [
                p for p in patterns.get('command_sequences', [])
                if p.get('confidence', 0) > 0.6
            ],
            'time_patterns': patterns.get('time_patterns', {}),
            'file_patterns': self._analyze_file_patterns(
                patterns.get('file_access', {})
            ),
            'error_patterns': patterns.get('error_patterns', [])
        }

        return significant_patterns

    def _consolidate_habit_knowledge(self) -> Dict[str, Any]:
        """Consolidate habit learning knowledge."""
        if not self.habit_learner:
            return {}

        habits = self.habit_learner.learned_habits

        # Get automation suggestions
        automations = self.habit_learner.generate_automations()

        return {
            'workflows': habits.get('workflow_habits', []),
            'routines': habits.get('daily_routines', []),
            'preferences': habits.get('tool_preferences', {}),
            'automations': automations,
            'shortcuts': habits.get('shortcuts', {})
        }

    def _consolidate_predictive_knowledge(self) -> Dict[str, Any]:
        """Consolidate predictive model knowledge."""
        if not self.predictive_model:
            return {}

        summary = self.predictive_model.get_prediction_summary()

        return {
            'prediction_accuracy': summary.get('overall_accuracy', 0),
            'model_performance': summary.get('model_accuracies', {}),
            'active_predictions': summary.get('active_predictions', 0)
        }

    def _merge_knowledge(self, *knowledge_sources):
        """Merge knowledge from multiple sources."""
        for source in knowledge_sources:
            for category, content in source.items():
                if isinstance(content, dict):
                    if category not in self.knowledge_base:
                        self.knowledge_base[category] = {}
                    if isinstance(self.knowledge_base[category], dict):
                        self.knowledge_base[category].update(content)
                    else:
                        # If existing is not a dict, replace it
                        self.knowledge_base[category] = content
                elif isinstance(content, list):
                    if category not in self.knowledge_base:
                        self.knowledge_base[category] = []
                    if isinstance(self.knowledge_base[category], list):
                        self.knowledge_base[category].extend(content)
                    else:
                        # If existing is not a list, replace it
                        self.knowledge_base[category] = content
                else:
                    self.knowledge_base[category] = content

    def _extract_relationships(self):
        """Extract relationships between knowledge entities."""
        # File-to-project relationships
        if 'projects' in self.knowledge_base:
            for project in self.knowledge_base['projects']:
                project_path = project.get('path', '')
                self.knowledge_graph.add_node(project_path, 'project')

                # Link files to projects
                for file_path in self.knowledge_base.get('file_patterns', {}).keys():
                    if file_path.startswith(project_path):
                        self.knowledge_graph.add_edge(
                            file_path, project_path, 'belongs_to'
                        )

        # Command-to-tool relationships
        if 'workflows' in self.knowledge_base:
            for workflow in self.knowledge_base['workflows']:
                for command in workflow.get('commands', []):
                    tool = command.split()[0] if command else ''
                    if tool:
                        self.knowledge_graph.add_edge(
                            workflow.get('name', 'unknown'),
                            tool,
                            'uses_tool'
                        )

        # Time-to-activity relationships
        if 'routines' in self.knowledge_base:
            for routine in self.knowledge_base['routines']:
                hour = routine.get('hour')
                activity = routine.get('activity')
                if hour and activity:
                    self.knowledge_graph.add_edge(
                        f"hour_{hour}",
                        activity,
                        'triggers'
                    )

    def _generate_insights(self) -> List[Dict[str, Any]]:
        """Generate high-level insights from consolidated knowledge."""
        insights = []

        # Workflow optimization insights
        if 'workflows' in self.knowledge_base:
            for workflow in self.knowledge_base['workflows']:
                if workflow.get('frequency', 0) > 10:
                    insights.append({
                        'type': 'automation_opportunity',
                        'description': f"Workflow '{workflow.get('name')}' executed {workflow.get('frequency')} times",
                        'recommendation': "Create permanent automation",
                        'importance': min(1.0, workflow.get('frequency', 0) / 20),
                        'actionable': True
                    })

        # Resource usage insights
        if 'system' in self.knowledge_base:
            hardware = self.knowledge_base['system'].get('hardware', {})
            if hardware.get('memory_usage_percent', 0) > 80:
                insights.append({
                    'type': 'resource_optimization',
                    'description': "High memory usage detected",
                    'recommendation': "Consider optimizing memory-intensive processes",
                    'importance': 0.8,
                    'actionable': True
                })

        # Error pattern insights
        if 'error_patterns' in self.knowledge_base:
            for error in self.knowledge_base['error_patterns']:
                if error.get('occurrences', 0) > 5:
                    insights.append({
                        'type': 'error_prevention',
                        'description': f"Recurring error: {error.get('type')}",
                        'recommendation': error.get('prevention', 'Investigate root cause'),
                        'importance': min(1.0, error.get('occurrences', 0) / 10),
                        'actionable': True
                    })

        # Tool preference insights
        if 'preferences' in self.knowledge_base:
            for tool, prefs in self.knowledge_base['preferences'].items():
                if prefs.get('frequency', 0) > 20:
                    insights.append({
                        'type': 'tool_optimization',
                        'description': f"Heavy usage of {tool}",
                        'recommendation': f"Optimize {tool} configuration for: {', '.join(prefs.get('common_contexts', [])[:3])}",
                        'importance': 0.6,
                        'actionable': True
                    })

        self.knowledge_base['insights'] = insights
        return insights

    def _optimize_knowledge_base(self):
        """Optimize knowledge base by removing redundancy and updating relevance."""
        # Remove duplicate workflows
        if 'workflows' in self.knowledge_base:
            seen_commands = set()
            unique_workflows = []

            for workflow in self.knowledge_base['workflows']:
                cmd_str = '|'.join(workflow.get('commands', []))
                if cmd_str not in seen_commands:
                    seen_commands.add(cmd_str)
                    unique_workflows.append(workflow)

            self.knowledge_base['workflows'] = unique_workflows

        # Update relevance scores based on recency
        now = datetime.now()

        for category in self.knowledge_base.values():
            if isinstance(category, list):
                for item in category:
                    if isinstance(item, dict) and 'last_seen' in item:
                        last_seen = datetime.fromisoformat(item['last_seen'])
                        days_old = (now - last_seen).days

                        # Decay relevance over time
                        if 'confidence' in item:
                            decay_factor = max(0.5, 1.0 - (days_old * 0.01))
                            item['relevance'] = item['confidence'] * decay_factor

    def _calculate_quality_metrics(self):
        """Calculate knowledge quality metrics."""
        total_items = self._count_knowledge_items()

        if total_items == 0:
            return

        # Accuracy: Based on prediction accuracy
        if 'prediction_accuracy' in self.knowledge_base:
            self.quality_metrics['accuracy'] = self.knowledge_base['prediction_accuracy']

        # Completeness: Coverage of system aspects
        covered_aspects = sum(1 for k, v in self.knowledge_base.items()
                            if v and k in ['system', 'projects', 'workflows'])
        self.quality_metrics['completeness'] = covered_aspects / 5  # Out of 5 main aspects

        # Consistency: Check for conflicts
        conflicts = self._check_consistency()
        self.quality_metrics['consistency'] = 1.0 - (conflicts / max(1, total_items))

        # Relevance: Based on recency and usage
        relevant_items = sum(1 for category in self.knowledge_base.values()
                           if isinstance(category, list)
                           for item in category
                           if isinstance(item, dict) and item.get('relevance', 0) > 0.5)
        self.quality_metrics['relevance'] = relevant_items / max(1, total_items)

        # Timeliness: Based on update recency
        recent_items = sum(1 for category in self.knowledge_base.values()
                         if isinstance(category, list)
                         for item in category
                         if isinstance(item, dict) and 'last_seen' in item
                         and (datetime.now() - datetime.fromisoformat(item['last_seen'])).days < 7)
        self.quality_metrics['timeliness'] = recent_items / max(1, total_items)

    def _check_consistency(self) -> int:
        """Check for consistency conflicts in knowledge base."""
        conflicts = 0

        # Check for conflicting workflows
        if 'workflows' in self.knowledge_base:
            workflows = self.knowledge_base['workflows']
            for i, w1 in enumerate(workflows):
                for w2 in workflows[i+1:]:
                    if self._workflows_conflict(w1, w2):
                        conflicts += 1

        return conflicts

    def _workflows_conflict(self, w1: Dict, w2: Dict) -> bool:
        """Check if two workflows conflict."""
        # Simple check: same starting commands but different continuations
        if not w1.get('commands') or not w2.get('commands'):
            return False

        if len(w1['commands']) > 2 and len(w2['commands']) > 2:
            if w1['commands'][:2] == w2['commands'][:2]:
                if w1['commands'][2:] != w2['commands'][2:]:
                    return True

        return False

    def _count_knowledge_items(self) -> int:
        """Count total knowledge items."""
        count = 0

        for category in self.knowledge_base.values():
            if isinstance(category, list):
                count += len(category)
            elif isinstance(category, dict):
                count += len(category)

        return count

    def _persist_knowledge(self):
        """Persist knowledge to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Store knowledge items
        for category, content in self.knowledge_base.items():
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        self._store_knowledge_item(cursor, category, item)
            elif isinstance(content, dict):
                for key, value in content.items():
                    self._store_knowledge_item(cursor, category, {key: value})

        # Store relationships
        for edge in self.knowledge_graph.edges:
            cursor.execute('''
                INSERT OR REPLACE INTO relationships (id, entity1, entity2, relationship_type, strength, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                self._generate_id(f"{edge['from']}{edge['to']}{edge['type']}"),
                edge['from'],
                edge['to'],
                edge['type'],
                edge.get('strength', 1.0),
                datetime.now()
            ))

        # Store insights
        for insight in self.knowledge_base.get('insights', []):
            cursor.execute('''
                INSERT OR REPLACE INTO insights (id, insight, category, importance, actionable, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                self._generate_id(insight['description']),
                insight['description'],
                insight['type'],
                insight.get('importance', 0.5),
                insight.get('actionable', False),
                datetime.now()
            ))

        conn.commit()
        conn.close()

    def _store_knowledge_item(self, cursor, category: str, item: Dict):
        """Store a knowledge item in database."""
        item_id = self._generate_id(f"{category}{json.dumps(item, sort_keys=True)}")

        cursor.execute('''
            INSERT OR REPLACE INTO knowledge_items
            (id, category, type, content, confidence, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item_id,
            category,
            item.get('type', 'general'),
            json.dumps(item),
            item.get('confidence', 0.5),
            item.get('source', 'unknown'),
            datetime.now(),
            datetime.now()
        ))

    def _analyze_file_patterns(self, file_access: Dict) -> Dict:
        """Analyze file access patterns for insights."""
        patterns = {}

        for file_path, access_data in file_access.items():
            if access_data.get('frequency', 0) > 3:
                file_type = Path(file_path).suffix
                if file_type not in patterns:
                    patterns[file_type] = {
                        'count': 0,
                        'total_accesses': 0,
                        'files': []
                    }

                patterns[file_type]['count'] += 1
                patterns[file_type]['total_accesses'] += access_data['frequency']
                patterns[file_type]['files'].append(file_path)

        return patterns

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for content."""
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def query_knowledge(self, query: str, category: str = None) -> List[Dict]:
        """
        Query the knowledge base.

        Args:
            query: Search query
            category: Optional category filter

        Returns:
            Matching knowledge items
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if category:
            cursor.execute('''
                SELECT content FROM knowledge_items
                WHERE category = ? AND content LIKE ?
                ORDER BY confidence DESC, updated_at DESC
                LIMIT 10
            ''', (category, f'%{query}%'))
        else:
            cursor.execute('''
                SELECT content FROM knowledge_items
                WHERE content LIKE ?
                ORDER BY confidence DESC, updated_at DESC
                LIMIT 10
            ''', (f'%{query}%',))

        results = []
        for row in cursor.fetchall():
            results.append(json.loads(row[0]))

        conn.close()
        return results

    def get_actionable_insights(self) -> List[Dict]:
        """Get actionable insights."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT insight, category, importance, created_at
            FROM insights
            WHERE actionable = 1 AND acted_upon = 0
            ORDER BY importance DESC
            LIMIT 5
        ''')

        insights = []
        for row in cursor.fetchall():
            insights.append({
                'insight': row[0],
                'category': row[1],
                'importance': row[2],
                'created_at': row[3]
            })

        conn.close()
        return insights

    def _load_consolidation_rules(self) -> Dict:
        """Load rules for knowledge consolidation."""
        return {
            'merge_similar': True,
            'remove_outdated': True,
            'enhance_relationships': True,
            'generate_insights': True
        }

    def export_knowledge(self, file_path: Path):
        """Export knowledge base to file."""
        export_data = {
            'knowledge_base': self.knowledge_base,
            'knowledge_graph': {
                'nodes': list(self.knowledge_graph.nodes),
                'edges': self.knowledge_graph.edges
            },
            'quality_metrics': self.quality_metrics,
            'export_time': datetime.now().isoformat()
        }

        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(f"Knowledge exported to {file_path}")


class KnowledgeGraph:
    """Simple knowledge graph implementation."""

    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node_id: str, node_type: str):
        """Add a node to the graph."""
        self.nodes[node_id] = {'type': node_type}

    def add_edge(self, from_node: str, to_node: str, edge_type: str):
        """Add an edge to the graph."""
        self.edges.append({
            'from': from_node,
            'to': to_node,
            'type': edge_type
        })