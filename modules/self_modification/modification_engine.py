#!/usr/bin/env python3
"""
Modification Engine Module - Generates code improvements using AI.
Capable of refactoring, optimization, and feature addition.
"""

import ast
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib


class ModificationEngine:
    """
    Generates concrete code modifications from improvement opportunities.
    """

    def __init__(self, llm_client=None, memory=None):
        """
        Initialize the Modification Engine.

        Args:
            llm_client: LLM client for code generation
            memory: Memory module for learning from past modifications
        """
        self.llm_client = llm_client
        self.memory = memory
        self.logger = logging.getLogger("PHOENIX.ModificationEngine")

        # Modification strategies
        self.strategies = {
            'optimization': self._optimize_code,
            'refactor': self._refactor_code,
            'bugfix': self._fix_bug,
            'feature': self._add_feature,
            'documentation': self._improve_documentation,
            'reliability': self._add_error_handling
        }

        # Track success patterns
        self.success_patterns = []
        if memory:
            self._load_success_patterns()

    async def generate_improvement(self, opportunity: Dict) -> Optional[Any]:
        """
        Generate a concrete improvement from an opportunity.

        Args:
            opportunity: The improvement opportunity

        Returns:
            Improvement object or None
        """
        from .self_improver import Improvement

        imp_type = opportunity.get('type', 'optimization')
        strategy = self.strategies.get(imp_type, self._generic_improvement)

        try:
            # Generate the improvement
            current_code, proposed_code, rationale = await strategy(opportunity)

            if not proposed_code or proposed_code == current_code:
                return None

            # Create improvement object
            improvement = Improvement(
                id=hashlib.md5(f"{opportunity}{datetime.now()}".encode()).hexdigest()[:12],
                type=imp_type,
                module=opportunity.get('module', 'unknown'),
                description=opportunity.get('description', ''),
                current_code=current_code,
                proposed_code=proposed_code,
                confidence=self._calculate_confidence(opportunity),
                impact=opportunity.get('priority', 'medium'),
                risk=self._assess_risk(opportunity),
                rationale=rationale
            )

            return improvement

        except Exception as e:
            self.logger.error(f"Failed to generate improvement: {e}")
            return None

    async def _optimize_code(self, opportunity: Dict) -> Tuple[str, str, str]:
        """
        Generate optimization improvements.

        Args:
            opportunity: The optimization opportunity

        Returns:
            Tuple of (current_code, proposed_code, rationale)
        """
        if not self.llm_client:
            return "", "", "LLM not available"

        prompt = f"""Optimize this Python code for better performance.
Opportunity: {opportunity['description']}
Location: {opportunity.get('location', 'unknown')}

Generate ONLY the optimized code that directly replaces the problematic code.
Focus on: performance, efficiency, readability.
Output format:
CURRENT:
<exact current code to replace>
PROPOSED:
<optimized replacement code>
RATIONALE:
<brief explanation>"""

        response = self.llm_client.generate(
            model='qwen2.5:14b-instruct',
            prompt=prompt,
            options={'temperature': 0.3}
        )

        return self._parse_modification_response(response['response'])

    async def _refactor_code(self, opportunity: Dict) -> Tuple[str, str, str]:
        """Generate refactoring improvements."""
        if not self.llm_client:
            return "", "", "LLM not available"

        prompt = f"""Refactor this Python code for better maintainability.
Issue: {opportunity['description']}

Generate a refactored version that:
1. Reduces complexity
2. Improves readability
3. Follows Python best practices

Output format:
CURRENT:
<exact current code>
PROPOSED:
<refactored code>
RATIONALE:
<explanation>"""

        response = self.llm_client.generate(
            model='qwen2.5:14b-instruct',
            prompt=prompt,
            options={'temperature': 0.3}
        )

        return self._parse_modification_response(response['response'])

    async def _add_error_handling(self, opportunity: Dict) -> Tuple[str, str, str]:
        """Add error handling to code."""
        if not self.llm_client:
            return "", "", "LLM not available"

        prompt = f"""Add proper error handling to this Python function.
Function: {opportunity.get('location', 'unknown')}
Issue: {opportunity['description']}

Add try/except blocks and proper error handling.
Output format:
CURRENT:
<function without error handling>
PROPOSED:
<function with error handling>
RATIONALE:
<what errors are handled and why>"""

        response = self.llm_client.generate(
            model='qwen2.5:14b-instruct',
            prompt=prompt,
            options={'temperature': 0.2}
        )

        return self._parse_modification_response(response['response'])

    async def _improve_documentation(self, opportunity: Dict) -> Tuple[str, str, str]:
        """Improve code documentation."""
        if not self.llm_client:
            return "", "", "LLM not available"

        prompt = f"""Add comprehensive docstrings and comments.
Location: {opportunity.get('location', 'unknown')}
Issue: {opportunity['description']}

Add proper docstrings following Google style.
Output format:
CURRENT:
<undocumented code>
PROPOSED:
<documented code>
RATIONALE:
<documentation improvements made>"""

        response = self.llm_client.generate(
            model='qwen2.5:14b-instruct',
            prompt=prompt,
            options={'temperature': 0.2}
        )

        return self._parse_modification_response(response['response'])

    async def _fix_bug(self, opportunity: Dict) -> Tuple[str, str, str]:
        """Fix identified bugs."""
        return await self._generic_improvement(opportunity)

    async def _add_feature(self, opportunity: Dict) -> Tuple[str, str, str]:
        """Add new features."""
        return await self._generic_improvement(opportunity)

    async def _generic_improvement(self, opportunity: Dict) -> Tuple[str, str, str]:
        """Generic improvement generation."""
        if not self.llm_client:
            return "", "", "LLM not available"

        prompt = f"""Improve this Python code.
Type: {opportunity.get('type', 'improvement')}
Description: {opportunity['description']}
Priority: {opportunity.get('priority', 'medium')}

Generate the improved version.
Output format:
CURRENT:
<current code>
PROPOSED:
<improved code>
RATIONALE:
<explanation>"""

        response = self.llm_client.generate(
            model='qwen2.5:14b-instruct',
            prompt=prompt,
            options={'temperature': 0.3}
        )

        return self._parse_modification_response(response['response'])

    def _parse_modification_response(self, response: str) -> Tuple[str, str, str]:
        """
        Parse LLM response into current, proposed, and rationale.

        Args:
            response: LLM response text

        Returns:
            Tuple of (current_code, proposed_code, rationale)
        """
        current = ""
        proposed = ""
        rationale = ""

        lines = response.split('\n')
        section = None

        for line in lines:
            if 'CURRENT:' in line.upper():
                section = 'current'
            elif 'PROPOSED:' in line.upper():
                section = 'proposed'
            elif 'RATIONALE:' in line.upper():
                section = 'rationale'
            elif section:
                if section == 'current':
                    current += line + '\n'
                elif section == 'proposed':
                    proposed += line + '\n'
                elif section == 'rationale':
                    rationale += line + '\n'

        return current.strip(), proposed.strip(), rationale.strip()

    def _calculate_confidence(self, opportunity: Dict) -> float:
        """Calculate confidence in the improvement."""
        confidence = 0.5  # Base confidence

        # Adjust based on priority
        if opportunity.get('priority') == 'high':
            confidence += 0.2
        elif opportunity.get('priority') == 'low':
            confidence -= 0.1

        # Adjust based on type
        if opportunity.get('type') in ['documentation', 'refactor']:
            confidence += 0.1  # These are usually safer

        # Check if similar improvements succeeded before
        if self._has_successful_pattern(opportunity):
            confidence += 0.2

        return min(confidence, 0.95)

    def _assess_risk(self, opportunity: Dict) -> str:
        """Assess risk level of the improvement."""
        risk_score = 0

        # High risk types
        if opportunity.get('type') in ['feature', 'architecture']:
            risk_score += 3

        # Medium risk types
        elif opportunity.get('type') in ['optimization', 'refactor']:
            risk_score += 2

        # Low risk types
        elif opportunity.get('type') in ['documentation', 'formatting']:
            risk_score += 1

        # Adjust for module criticality
        if 'core' in opportunity.get('module', ''):
            risk_score += 2

        if risk_score >= 4:
            return 'high'
        elif risk_score >= 2:
            return 'medium'
        return 'low'

    def _has_successful_pattern(self, opportunity: Dict) -> bool:
        """Check if similar improvements succeeded before."""
        for pattern in self.success_patterns:
            if (pattern.get('type') == opportunity.get('type') and
                pattern.get('module') == opportunity.get('module')):
                return True
        return False

    def _load_success_patterns(self):
        """Load successful patterns from memory."""
        if self.memory:
            # Load from memory
            pass  # TODO: Implement memory loading

    def learn_from_result(self, improvement: Any, success: bool):
        """
        Learn from the result of an improvement attempt.

        Args:
            improvement: The improvement that was attempted
            success: Whether it was successful
        """
        if success:
            self.success_patterns.append({
                'type': improvement.type,
                'module': improvement.module,
                'confidence': improvement.confidence
            })

            if self.memory:
                self.memory.learn_fact(
                    f"Successful {improvement.type} improvement in {improvement.module}",
                    'modification_patterns'
                )

