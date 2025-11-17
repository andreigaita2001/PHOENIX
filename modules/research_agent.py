#!/usr/bin/env python3
"""
Research Agent Module - Autonomous, thorough online research with learning capabilities.
This module conducts deep research, synthesizes information, and improves from feedback.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import time

class ResearchAgent:
    """
    Autonomous research agent that conducts thorough online research,
    learns from feedback, and continuously improves its methodology.
    """

    def __init__(self, web_browser=None, memory_manager=None, llm_client=None):
        """
        Initialize the Research Agent.

        Args:
            web_browser: Web browser module for accessing internet
            memory_manager: Memory system for storing research and learning
            llm_client: LLM for analysis and synthesis
        """
        self.web_browser = web_browser
        self.memory_manager = memory_manager
        self.llm_client = llm_client
        self.logger = logging.getLogger("PHOENIX.ResearchAgent")

        # Research state
        self.current_research = {
            'topic': None,
            'objective': None,
            'sources': [],
            'findings': [],
            'synthesis': None,
            'confidence': 0.0,
            'depth': 0,
            'start_time': None
        }

        # Research methodology (learns and improves)
        self.methodology = {
            'max_depth': 3,  # How many levels deep to research
            'min_sources': 5,  # Minimum sources to consult
            'max_sources': 20,  # Maximum sources to prevent endless research
            'quality_threshold': 0.7,  # Minimum quality score for sources
            'synthesis_approach': 'comparative',  # comparative, comprehensive, focused
            'verification_required': True,  # Cross-check facts
            'time_limit': 600  # 10 minutes max per research
        }

        # Learning from feedback
        self.feedback_history = []
        self.research_history = []
        self.improvement_suggestions = []

        # Load saved methodology improvements
        self._load_methodology()

        self.logger.info("Research Agent initialized")

    def research(self, topic: str, objective: str = None, requirements: Dict = None) -> Dict[str, Any]:
        """
        Conduct thorough research on a topic.

        Args:
            topic: The topic to research
            objective: Specific objective (e.g., "find the best implementation")
            requirements: Specific requirements for the research

        Returns:
            Research results with synthesis and recommendations
        """
        self.logger.info(f"Starting research on: {topic}")

        # Initialize research session
        self.current_research = {
            'topic': topic,
            'objective': objective or f"Comprehensive research on {topic}",
            'sources': [],
            'findings': [],
            'synthesis': None,
            'confidence': 0.0,
            'depth': 0,
            'start_time': datetime.now(),
            'requirements': requirements or {}
        }

        # Phase 1: Initial Search and Source Discovery
        self.logger.info("Phase 1: Source Discovery")
        sources = self._discover_sources(topic)

        # Phase 2: Deep Dive into Quality Sources
        self.logger.info("Phase 2: Deep Dive")
        findings = self._deep_dive(sources)

        # Phase 3: Fact Verification and Cross-Referencing
        if self.methodology['verification_required']:
            self.logger.info("Phase 3: Verification")
            findings = self._verify_findings(findings)

        # Phase 4: Synthesis and Analysis
        self.logger.info("Phase 4: Synthesis")
        synthesis = self._synthesize_findings(findings)

        # Phase 5: Generate Recommendations
        self.logger.info("Phase 5: Recommendations")
        recommendations = self._generate_recommendations(synthesis)

        # Compile final results
        results = {
            'topic': topic,
            'objective': objective,
            'sources_consulted': len(self.current_research['sources']),
            'confidence': self._calculate_confidence(),
            'synthesis': synthesis,
            'recommendations': recommendations,
            'key_findings': findings[:10],  # Top 10 findings
            'research_trail': self._get_research_trail(),
            'time_taken': (datetime.now() - self.current_research['start_time']).seconds
        }

        # Store in history
        self.research_history.append(results)

        # Learn from this research
        self._learn_from_research(results)

        # Save to memory if available
        if self.memory_manager:
            self.memory_manager.learn_fact(
                f"Research on {topic}: {synthesis[:200]}",
                category='research'
            )

        return results

    def _discover_sources(self, topic: str) -> List[Dict]:
        """Discover quality sources for the topic."""
        sources = []

        if not self.web_browser:
            self.logger.error("No web browser available for research")
            return sources

        # Search with different queries for comprehensive coverage
        search_queries = [
            topic,
            f"{topic} best practices",
            f"{topic} implementation guide",
            f"{topic} comparison",
            f"{topic} tutorial",
            f"{topic} documentation"
        ]

        for query in search_queries[:3]:  # Start with first 3
            try:
                search_results = self.web_browser.search(query)
                if search_results.get('success'):
                    for result in search_results.get('results', []):
                        # Score the source quality
                        quality_score = self._score_source_quality(result)

                        if quality_score >= self.methodology['quality_threshold']:
                            sources.append({
                                'url': result.get('url'),
                                'title': result.get('title'),
                                'snippet': result.get('snippet'),
                                'content': result.get('content'),
                                'quality_score': quality_score,
                                'query': query
                            })

                        if len(sources) >= self.methodology['max_sources']:
                            break

            except Exception as e:
                self.logger.error(f"Error searching for {query}: {e}")

            if len(sources) >= self.methodology['min_sources']:
                break

        self.current_research['sources'] = sources
        return sources

    def _deep_dive(self, sources: List[Dict]) -> List[Dict]:
        """Deep dive into sources to extract detailed information."""
        findings = []

        for source in sources[:self.methodology['max_sources']]:
            try:
                # Navigate to the source
                if self.web_browser and source.get('url'):
                    content = self.web_browser.navigate(source['url'])

                    if content.get('success'):
                        # Extract key information
                        finding = {
                            'source': source['url'],
                            'title': source['title'],
                            'content': content.get('text', ''),
                            'key_points': self._extract_key_points(content.get('text', '')),
                            'relevance': self._calculate_relevance(
                                content.get('text', ''),
                                self.current_research['topic']
                            )
                        }
                        findings.append(finding)

                        # Follow relevant links if depth allows
                        if self.current_research['depth'] < self.methodology['max_depth']:
                            self._follow_relevant_links(content.get('links', []))

            except Exception as e:
                self.logger.error(f"Error diving into {source.get('url')}: {e}")

        self.current_research['findings'] = findings
        return findings

    def _verify_findings(self, findings: List[Dict]) -> List[Dict]:
        """Cross-reference and verify findings."""
        verified_findings = []

        for finding in findings:
            # Check if key points appear in multiple sources
            confirmation_count = 0

            for other_finding in findings:
                if finding != other_finding:
                    for point in finding.get('key_points', []):
                        if self._point_appears_in(point, other_finding.get('content', '')):
                            confirmation_count += 1

            finding['verification_score'] = confirmation_count / max(len(findings) - 1, 1)
            verified_findings.append(finding)

        return verified_findings

    def _synthesize_findings(self, findings: List[Dict]) -> str:
        """Synthesize all findings into a coherent summary."""
        if not findings:
            return "No significant findings from research."

        # Group findings by theme
        themes = self._identify_themes(findings)

        # Build synthesis based on methodology
        if self.methodology['synthesis_approach'] == 'comparative':
            synthesis = self._comparative_synthesis(themes)
        elif self.methodology['synthesis_approach'] == 'comprehensive':
            synthesis = self._comprehensive_synthesis(themes)
        else:
            synthesis = self._focused_synthesis(themes)

        self.current_research['synthesis'] = synthesis
        return synthesis

    def _generate_recommendations(self, synthesis: str) -> List[Dict]:
        """Generate actionable recommendations based on research."""
        recommendations = []

        # Analyze synthesis for actionable items
        if self.current_research.get('objective'):
            if 'implement' in self.current_research['objective'].lower():
                recommendations.append({
                    'type': 'implementation',
                    'priority': 'high',
                    'recommendation': 'Based on research, implement using the most common approach found',
                    'confidence': self.current_research.get('confidence', 0.5)
                })

            if 'best' in self.current_research['objective'].lower():
                recommendations.append({
                    'type': 'best_practice',
                    'priority': 'high',
                    'recommendation': 'Follow the most frequently recommended practices',
                    'confidence': self.current_research.get('confidence', 0.5)
                })

        return recommendations

    def provide_feedback(self, research_id: str, feedback: Dict) -> Dict:
        """
        Receive feedback on research quality and learn from it.

        Args:
            research_id: ID of the research session
            feedback: Feedback dictionary with ratings and comments

        Returns:
            Acknowledgment and improvements made
        """
        self.feedback_history.append({
            'research_id': research_id,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        })

        # Analyze feedback for improvements
        improvements = self._analyze_feedback(feedback)

        # Apply improvements to methodology
        for improvement in improvements:
            self._apply_improvement(improvement)

        # Save updated methodology
        self._save_methodology()

        return {
            'feedback_received': True,
            'improvements_applied': improvements,
            'current_methodology': self.methodology
        }

    def _analyze_feedback(self, feedback: Dict) -> List[Dict]:
        """Analyze feedback to identify improvements."""
        improvements = []

        # Check feedback ratings
        if feedback.get('depth_rating', 0) < 3:
            improvements.append({
                'type': 'depth',
                'action': 'increase',
                'reason': 'User indicated research wasn\'t deep enough'
            })

        if feedback.get('source_quality', 0) < 3:
            improvements.append({
                'type': 'quality_threshold',
                'action': 'increase',
                'reason': 'User indicated source quality was poor'
            })

        if feedback.get('too_long', False):
            improvements.append({
                'type': 'max_sources',
                'action': 'decrease',
                'reason': 'User indicated research took too long'
            })

        # Learn from specific comments
        if feedback.get('comments'):
            # This would use NLP to extract specific improvements
            # For now, just log them
            self.improvement_suggestions.append(feedback['comments'])

        return improvements

    def _apply_improvement(self, improvement: Dict):
        """Apply a specific improvement to methodology."""
        if improvement['type'] == 'depth' and improvement['action'] == 'increase':
            self.methodology['max_depth'] = min(self.methodology['max_depth'] + 1, 5)

        elif improvement['type'] == 'quality_threshold' and improvement['action'] == 'increase':
            self.methodology['quality_threshold'] = min(
                self.methodology['quality_threshold'] + 0.05, 0.95
            )

        elif improvement['type'] == 'max_sources' and improvement['action'] == 'decrease':
            self.methodology['max_sources'] = max(self.methodology['max_sources'] - 2, 5)

        self.logger.info(f"Applied improvement: {improvement}")

    def get_research_status(self) -> Dict:
        """Get current research status."""
        if not self.current_research['topic']:
            return {'status': 'idle', 'message': 'No active research'}

        time_elapsed = (datetime.now() - self.current_research['start_time']).seconds

        return {
            'status': 'active',
            'topic': self.current_research['topic'],
            'sources_found': len(self.current_research['sources']),
            'findings': len(self.current_research['findings']),
            'time_elapsed': time_elapsed,
            'confidence': self.current_research['confidence']
        }

    def get_research_history(self, limit: int = 10) -> List[Dict]:
        """Get recent research history."""
        return self.research_history[-limit:]

    def export_research(self, research_id: str = None) -> Dict:
        """Export research results in a structured format."""
        if research_id:
            # Find specific research
            for research in self.research_history:
                if research.get('id') == research_id:
                    return research
        else:
            # Export current research
            return self.current_research

    # Helper methods
    def _score_source_quality(self, source: Dict) -> float:
        """Score source quality based on various factors."""
        score = 0.5  # Base score

        # Check for quality indicators
        url = source.get('url', '')
        if any(domain in url for domain in ['.edu', '.gov', 'github.com', 'stackoverflow.com']):
            score += 0.2
        if 'documentation' in url or 'docs' in url:
            score += 0.1
        if source.get('content'):
            score += 0.1

        return min(score, 1.0)

    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text."""
        # Simplified extraction - would use NLP in production
        lines = text.split('\n')
        key_points = []

        for line in lines:
            line = line.strip()
            if len(line) > 50 and len(line) < 200:
                if any(indicator in line.lower() for indicator in
                       ['important', 'key', 'must', 'should', 'best practice', 'recommend']):
                    key_points.append(line)

        return key_points[:10]  # Top 10 key points

    def _calculate_relevance(self, text: str, topic: str) -> float:
        """Calculate relevance of text to topic."""
        topic_words = topic.lower().split()
        text_lower = text.lower()

        relevance = sum(1 for word in topic_words if word in text_lower) / len(topic_words)
        return min(relevance, 1.0)

    def _point_appears_in(self, point: str, text: str) -> bool:
        """Check if a key point appears in text."""
        # Simplified check - would use semantic similarity in production
        key_words = point.lower().split()[:5]  # First 5 words
        return all(word in text.lower() for word in key_words)

    def _identify_themes(self, findings: List[Dict]) -> Dict[str, List]:
        """Identify common themes across findings."""
        themes = {}

        # Simplified theme identification
        for finding in findings:
            for point in finding.get('key_points', []):
                # Use first significant word as theme (simplified)
                words = point.split()
                for word in words:
                    if len(word) > 4:  # Significant word
                        if word not in themes:
                            themes[word] = []
                        themes[word].append(point)
                        break

        return themes

    def _comparative_synthesis(self, themes: Dict) -> str:
        """Create comparative synthesis."""
        synthesis = "Research reveals multiple approaches:\n\n"

        for theme, points in list(themes.items())[:5]:
            synthesis += f"**{theme.title()}:**\n"
            for point in points[:2]:
                synthesis += f"- {point}\n"
            synthesis += "\n"

        return synthesis

    def _comprehensive_synthesis(self, themes: Dict) -> str:
        """Create comprehensive synthesis."""
        synthesis = "Comprehensive research findings:\n\n"

        for theme, points in themes.items():
            synthesis += f"**{theme.title()} ({len(points)} mentions):**\n"
            for point in points[:3]:
                synthesis += f"- {point}\n"
            synthesis += "\n"

        return synthesis

    def _focused_synthesis(self, themes: Dict) -> str:
        """Create focused synthesis on most important themes."""
        # Sort themes by frequency
        sorted_themes = sorted(themes.items(), key=lambda x: len(x[1]), reverse=True)

        synthesis = "Key findings from research:\n\n"

        for theme, points in sorted_themes[:3]:  # Top 3 themes
            synthesis += f"**{theme.title()}** (High importance):\n"
            for point in points[:2]:
                synthesis += f"- {point}\n"
            synthesis += "\n"

        return synthesis

    def _calculate_confidence(self) -> float:
        """Calculate confidence in research results."""
        confidence = 0.0

        # Factors affecting confidence
        if len(self.current_research['sources']) >= self.methodology['min_sources']:
            confidence += 0.3
        if len(self.current_research['findings']) >= 3:
            confidence += 0.3
        if self.current_research.get('synthesis'):
            confidence += 0.2
        if self.methodology['verification_required']:
            confidence += 0.2

        self.current_research['confidence'] = confidence
        return confidence

    def _get_research_trail(self) -> List[str]:
        """Get the trail of research steps taken."""
        trail = []

        for source in self.current_research['sources'][:5]:
            trail.append(f"Searched: {source.get('query', 'Unknown')}")
            trail.append(f"Found: {source.get('title', 'Unknown')}")

        return trail

    def _follow_relevant_links(self, links: List[Dict]):
        """Follow relevant links to deepen research."""
        self.current_research['depth'] += 1

        # Select most relevant links
        relevant_links = []
        for link in links[:5]:  # Check first 5 links
            if any(keyword in link.get('text', '').lower()
                   for keyword in self.current_research['topic'].lower().split()):
                relevant_links.append(link)

        # Follow top relevant link
        if relevant_links and self.web_browser:
            try:
                top_link = relevant_links[0]
                content = self.web_browser.navigate(top_link['url'])
                if content.get('success'):
                    # Add to findings
                    self.current_research['findings'].append({
                        'source': top_link['url'],
                        'title': top_link.get('text', 'Linked page'),
                        'content': content.get('text', '')[:1000],
                        'depth': self.current_research['depth']
                    })
            except Exception as e:
                self.logger.error(f"Error following link: {e}")

    def _save_methodology(self):
        """Save current methodology to file."""
        try:
            methodology_file = Path('data/research_methodology.json')
            methodology_file.parent.mkdir(exist_ok=True)

            with open(methodology_file, 'w') as f:
                json.dump(self.methodology, f, indent=2)

            self.logger.info("Saved research methodology")
        except Exception as e:
            self.logger.error(f"Error saving methodology: {e}")

    def _load_methodology(self):
        """Load saved methodology if exists."""
        try:
            methodology_file = Path('data/research_methodology.json')

            if methodology_file.exists():
                with open(methodology_file, 'r') as f:
                    saved_methodology = json.load(f)
                    self.methodology.update(saved_methodology)

                self.logger.info("Loaded saved research methodology")
        except Exception as e:
            self.logger.error(f"Error loading methodology: {e}")

    def get_capabilities(self) -> Dict[str, bool]:
        """Return research capabilities."""
        return {
            'autonomous_research': True,
            'deep_dive': True,
            'source_verification': True,
            'synthesis': True,
            'learning_from_feedback': True,
            'methodology_improvement': True,
            'research_history': True,
            'export_results': True
        }