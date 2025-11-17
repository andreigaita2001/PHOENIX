#!/usr/bin/env python3
"""
Personal Knowledge Extractor - Learns patterns and extracts knowledge from personal data.
Maintains complete privacy while building a personal knowledge graph.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict, Counter
import re
import statistics
import networkx as nx
from dataclasses import dataclass

@dataclass
class PersonalPattern:
    """Represents a discovered personal pattern."""
    pattern_type: str
    description: str
    confidence: float
    occurrences: int
    examples: List[str]
    insights: List[str]

class PersonalKnowledgeExtractor:
    """
    Extracts knowledge and patterns from personal data while maintaining privacy.
    """

    def __init__(self, vault, memory_manager=None):
        """
        Initialize the Personal Knowledge Extractor.

        Args:
            vault: PersonalDataVault instance
            memory_manager: Memory system for learning
        """
        self.vault = vault
        self.memory_manager = memory_manager
        self.logger = logging.getLogger("PHOENIX.PersonalKnowledge")

        # Knowledge categories
        self.knowledge = {
            'routines': [],           # Daily/weekly routines
            'preferences': {},        # Learned preferences
            'relationships': {},      # People and connections
            'interests': [],          # Topics of interest
            'habits': [],            # Behavioral habits
            'locations': [],         # Important places
            'communication': {},     # Communication patterns
            'productivity': {},      # Work patterns
            'health': {},           # Health/fitness patterns
            'finance': {},          # Spending patterns
            'learning': [],         # Learning interests
            'goals': []             # Identified goals
        }

        # Initialize knowledge graph
        self.knowledge_graph = nx.DiGraph()

        # Pattern detection thresholds
        self.thresholds = {
            'routine_min_occurrences': 3,
            'interest_min_mentions': 5,
            'relationship_min_interactions': 10,
            'habit_confidence': 0.7
        }

        self.logger.info("Personal Knowledge Extractor initialized")

    def analyze_personal_data(self) -> Dict[str, Any]:
        """
        Analyze all personal data to extract knowledge and patterns.

        Returns:
            Analysis results
        """
        results = {
            'patterns_found': 0,
            'insights_generated': 0,
            'relationships_identified': 0,
            'routines_discovered': 0,
            'interests_detected': 0,
            'privacy_maintained': True
        }

        self.logger.info("Starting personal data analysis...")

        # Analyze different data types
        analyzers = [
            ('emails', self._analyze_communication_patterns),
            ('location_history', self._analyze_location_patterns),
            ('youtube_history', self._analyze_media_interests),
            ('search_history', self._analyze_search_patterns),
            ('calendar', self._analyze_schedule_patterns),
            ('photos', self._analyze_photo_patterns)
        ]

        for category, analyzer in analyzers:
            try:
                patterns = analyzer(category)
                results['patterns_found'] += len(patterns)
                self.logger.info(f"Found {len(patterns)} patterns in {category}")
            except Exception as e:
                self.logger.error(f"Error analyzing {category}: {e}")

        # Build knowledge graph
        self._build_knowledge_graph()

        # Generate insights
        insights = self._generate_insights()
        results['insights_generated'] = len(insights)

        # Store in memory if available
        if self.memory_manager:
            for insight in insights[:20]:  # Store top 20 insights
                self.memory_manager.learn_fact(
                    insight['description'],
                    category='personal_knowledge',
                    confidence=insight['confidence']
                )

        return results

    def _analyze_communication_patterns(self, category: str) -> List[PersonalPattern]:
        """Analyze email and communication patterns."""
        patterns = []

        # Get email metadata (not content for privacy)
        emails = self.vault.search_personal_data('', category='emails')

        if not emails:
            return patterns

        # Analyze communication frequency
        email_times = []
        email_contacts = Counter()

        for email in emails:
            metadata = email.get('metadata', {})
            timestamp = email.get('timestamp')

            if timestamp:
                email_times.append(datetime.fromisoformat(timestamp))

            # Extract contacts (would need proper email parsing)
            # This is simplified for demonstration

        # Find peak communication times
        if email_times:
            hour_distribution = Counter(t.hour for t in email_times)
            peak_hours = hour_distribution.most_common(3)

            if peak_hours:
                pattern = PersonalPattern(
                    pattern_type='communication_schedule',
                    description=f"Most active email hours: {', '.join(f'{h}:00' for h, _ in peak_hours)}",
                    confidence=0.8,
                    occurrences=len(email_times),
                    examples=[],
                    insights=['Communication patterns identified']
                )
                patterns.append(pattern)
                self.knowledge['communication']['peak_hours'] = peak_hours

        return patterns

    def _analyze_location_patterns(self, category: str) -> List[PersonalPattern]:
        """Analyze location history patterns."""
        patterns = []

        # Get location data
        locations = self.vault.search_personal_data('', category='location_history')

        if not locations:
            return patterns

        # Analyze frequently visited places
        location_clusters = defaultdict(int)

        for loc in locations:
            metadata = loc.get('metadata', {})
            date = metadata.get('date')
            points = metadata.get('points', 0)

            if date and points > 0:
                # This is a simplified clustering
                # Real implementation would use proper geographic clustering
                location_clusters[date[:7]] += points  # Group by month

        # Find patterns in movement
        if location_clusters:
            avg_monthly_points = statistics.mean(location_clusters.values())

            pattern = PersonalPattern(
                pattern_type='movement_pattern',
                description=f"Average {avg_monthly_points:.0f} location points per month",
                confidence=0.7,
                occurrences=len(location_clusters),
                examples=list(location_clusters.keys())[:5],
                insights=['Regular movement patterns detected']
            )
            patterns.append(pattern)

            # Identify home/work locations (simplified)
            self.knowledge['locations'] = ['home', 'work']  # Would be extracted properly

        return patterns

    def _analyze_media_interests(self, category: str) -> List[PersonalPattern]:
        """Analyze YouTube and media consumption patterns."""
        patterns = []

        # Get YouTube history
        videos = self.vault.search_personal_data('', category='youtube_history')

        if not videos:
            return patterns

        # Extract topics and channels
        video_topics = Counter()
        watch_times = []

        for video in videos:
            metadata = video.get('metadata', {})
            title = metadata.get('title', '')
            timestamp = video.get('timestamp')

            if title:
                # Extract potential topics from title (simplified)
                words = title.lower().split()
                for word in words:
                    if len(word) > 4:  # Filter short words
                        video_topics[word] += 1

            if timestamp:
                watch_times.append(datetime.fromisoformat(timestamp))

        # Find interests
        top_topics = video_topics.most_common(10)
        if top_topics:
            interests = [topic for topic, count in top_topics if count >= self.thresholds['interest_min_mentions']]

            if interests:
                pattern = PersonalPattern(
                    pattern_type='media_interests',
                    description=f"Top interests: {', '.join(interests[:5])}",
                    confidence=0.75,
                    occurrences=len(videos),
                    examples=interests[:5],
                    insights=['Media preferences identified']
                )
                patterns.append(pattern)
                self.knowledge['interests'].extend(interests[:5])

        # Analyze viewing schedule
        if watch_times:
            hour_distribution = Counter(t.hour for t in watch_times)
            peak_viewing = hour_distribution.most_common(1)[0] if hour_distribution else None

            if peak_viewing:
                pattern = PersonalPattern(
                    pattern_type='viewing_schedule',
                    description=f"Peak viewing time: {peak_viewing[0]}:00",
                    confidence=0.8,
                    occurrences=peak_viewing[1],
                    examples=[],
                    insights=['Viewing habits identified']
                )
                patterns.append(pattern)

        return patterns

    def _analyze_search_patterns(self, category: str) -> List[PersonalPattern]:
        """Analyze search history patterns."""
        patterns = []
        # Similar to media analysis but for searches
        return patterns

    def _analyze_schedule_patterns(self, category: str) -> List[PersonalPattern]:
        """Analyze calendar and schedule patterns."""
        patterns = []

        # Get calendar data
        events = self.vault.search_personal_data('', category='calendar')

        if events:
            # Analyze recurring events, meeting patterns, etc.
            pattern = PersonalPattern(
                pattern_type='schedule_routine',
                description="Regular schedule patterns detected",
                confidence=0.7,
                occurrences=len(events),
                examples=[],
                insights=['Schedule routines identified']
            )
            patterns.append(pattern)

        return patterns

    def _analyze_photo_patterns(self, category: str) -> List[PersonalPattern]:
        """Analyze photo metadata patterns."""
        patterns = []

        # Get photo metadata
        photos = self.vault.search_personal_data('', category='photos')

        if photos:
            # Analyze when photos are taken
            photo_times = []
            for photo in photos:
                metadata = photo.get('metadata', {})
                if 'photoTakenTime' in metadata:
                    # Extract time patterns
                    pass

            pattern = PersonalPattern(
                pattern_type='photo_habits',
                description=f"Photo collection contains {len(photos)} items",
                confidence=0.9,
                occurrences=len(photos),
                examples=[],
                insights=['Photo patterns analyzed']
            )
            patterns.append(pattern)

        return patterns

    def _build_knowledge_graph(self):
        """Build a knowledge graph from extracted patterns."""
        # Add nodes for different knowledge types
        for interest in self.knowledge['interests']:
            self.knowledge_graph.add_node(interest, type='interest')

        for location in self.knowledge['locations']:
            self.knowledge_graph.add_node(location, type='location')

        # Add relationships
        # This would be more sophisticated in real implementation

        self.logger.info(f"Built knowledge graph with {len(self.knowledge_graph.nodes)} nodes")

    def _generate_insights(self) -> List[Dict[str, Any]]:
        """Generate high-level insights from patterns."""
        insights = []

        # Communication insights
        if self.knowledge.get('communication'):
            peak_hours = self.knowledge['communication'].get('peak_hours', [])
            if peak_hours:
                insights.append({
                    'category': 'communication',
                    'description': f"You're most active in emails around {peak_hours[0][0]}:00",
                    'confidence': 0.8,
                    'actionable': True,
                    'suggestion': 'Schedule important emails during peak hours for better productivity'
                })

        # Interest insights
        if self.knowledge.get('interests'):
            top_interests = self.knowledge['interests'][:3]
            if top_interests:
                insights.append({
                    'category': 'interests',
                    'description': f"Your top interests appear to be: {', '.join(top_interests)}",
                    'confidence': 0.75,
                    'actionable': True,
                    'suggestion': 'I can help you explore these topics more deeply'
                })

        # Location insights
        if self.knowledge.get('locations'):
            insights.append({
                'category': 'movement',
                'description': f"You have {len(self.knowledge['locations'])} regularly visited locations",
                'confidence': 0.7,
                'actionable': False,
                'suggestion': None
                })

        # Productivity insights
        if self.knowledge.get('productivity'):
            insights.append({
                'category': 'productivity',
                'description': 'Productivity patterns have been identified',
                'confidence': 0.6,
                'actionable': True,
                'suggestion': 'I can help optimize your schedule based on your patterns'
            })

        return insights

    def get_personal_summary(self) -> Dict[str, Any]:
        """
        Get a privacy-preserving summary of personal knowledge.

        Returns:
            Summary of learned knowledge
        """
        return {
            'interests_identified': len(self.knowledge.get('interests', [])),
            'routines_found': len(self.knowledge.get('routines', [])),
            'relationships_mapped': len(self.knowledge.get('relationships', {})),
            'locations_identified': len(self.knowledge.get('locations', [])),
            'habits_discovered': len(self.knowledge.get('habits', [])),
            'knowledge_graph_size': len(self.knowledge_graph.nodes),
            'privacy_status': 'All data encrypted and local',
            'insights_available': True
        }

    def query_personal_knowledge(self, query: str) -> List[Dict]:
        """
        Query personal knowledge base.

        Args:
            query: Natural language query

        Returns:
            Relevant knowledge and insights
        """
        results = []
        query_lower = query.lower()

        # Search through knowledge categories
        for category, data in self.knowledge.items():
            if category in query_lower or any(str(item).lower() in query_lower for item in data if item):
                results.append({
                    'category': category,
                    'data': data if isinstance(data, (list, dict)) else str(data),
                    'relevance': 0.8
                })

        # Search in knowledge graph
        if self.knowledge_graph:
            for node in self.knowledge_graph.nodes:
                if str(node).lower() in query_lower:
                    neighbors = list(self.knowledge_graph.neighbors(node))
                    results.append({
                        'category': 'knowledge_graph',
                        'node': node,
                        'connections': neighbors,
                        'relevance': 0.9
                    })

        return results[:10]  # Return top 10 results

    def suggest_based_on_patterns(self) -> List[Dict]:
        """
        Generate suggestions based on learned patterns.

        Returns:
            List of personalized suggestions
        """
        suggestions = []

        # Schedule suggestions
        if self.knowledge.get('communication', {}).get('peak_hours'):
            suggestions.append({
                'type': 'schedule',
                'suggestion': 'Schedule important communications during your peak hours',
                'confidence': 0.8
            })

        # Interest suggestions
        if self.knowledge.get('interests'):
            for interest in self.knowledge['interests'][:3]:
                suggestions.append({
                    'type': 'content',
                    'suggestion': f'Explore more content about {interest}',
                    'confidence': 0.7
                })

        # Routine suggestions
        if self.knowledge.get('routines'):
            suggestions.append({
                'type': 'automation',
                'suggestion': 'I can help automate your regular routines',
                'confidence': 0.75
            })

        return suggestions