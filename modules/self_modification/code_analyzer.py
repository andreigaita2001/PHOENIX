#!/usr/bin/env python3
"""
Code Analyzer Module - Analyzes PHOENIX's codebase for improvement opportunities.
Uses AST parsing and AI reasoning to understand code structure and quality.
"""

import ast
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import importlib.util
import networkx as nx
from dataclasses import dataclass


@dataclass
class CodeMetrics:
    """Metrics for a code module."""
    lines_of_code: int
    complexity: int
    function_count: int
    class_count: int
    import_count: int
    docstring_coverage: float
    comment_ratio: float
    max_function_complexity: int
    duplicate_code_blocks: int
    unused_imports: List[str]
    unused_variables: List[str]


class CodeAnalyzer:
    """
    Analyzes PHOENIX's codebase to find improvement opportunities.
    """

    def __init__(self, phoenix_root: Path, llm_client=None):
        """
        Initialize the Code Analyzer.

        Args:
            phoenix_root: Root directory of PHOENIX
            llm_client: LLM client for advanced analysis
        """
        self.phoenix_root = phoenix_root
        self.llm_client = llm_client
        self.logger = logging.getLogger("PHOENIX.CodeAnalyzer")

        # Analysis patterns
        self.improvement_patterns = {
            'performance': [
                'nested loops with O(n²) complexity',
                'repeated file I/O in loops',
                'unnecessary list comprehensions',
                'missing caching opportunities',
                'synchronous code that could be async'
            ],
            'maintainability': [
                'functions longer than 50 lines',
                'classes with too many responsibilities',
                'missing docstrings',
                'complex conditional logic',
                'hardcoded values that should be configurable'
            ],
            'reliability': [
                'missing error handling',
                'unclosed resources',
                'race conditions',
                'missing input validation',
                'potential null/None references'
            ],
            'architecture': [
                'circular dependencies',
                'tight coupling',
                'missing abstraction layers',
                'violation of single responsibility',
                'inconsistent naming conventions'
            ]
        }

        self.dependency_graph = nx.DiGraph()

    async def analyze_module(self, module_path: Path) -> Dict[str, Any]:
        """
        Analyze a single Python module.

        Args:
            module_path: Path to the module file

        Returns:
            Analysis results
        """
        self.logger.info(f"Analyzing module: {module_path}")

        with open(module_path, 'r') as f:
            source_code = f.read()

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            self.logger.error(f"Syntax error in {module_path}: {e}")
            return {'error': str(e)}

        # Collect metrics
        metrics = self._calculate_metrics(tree, source_code)

        # Find improvement opportunities
        opportunities = await self._find_opportunities(tree, source_code, module_path)

        # Analyze dependencies
        dependencies = self._extract_dependencies(tree)

        # AI-powered deep analysis if available
        if self.llm_client:
            ai_insights = await self._ai_analysis(source_code, module_path.stem)
            opportunities.extend(ai_insights)

        return {
            'module': str(module_path.relative_to(self.phoenix_root)),
            'metrics': metrics.__dict__,
            'opportunities': opportunities,
            'dependencies': dependencies
        }

    def _calculate_metrics(self, tree: ast.AST, source_code: str) -> CodeMetrics:
        """
        Calculate code metrics for a module.

        Args:
            tree: AST of the module
            source_code: Source code string

        Returns:
            CodeMetrics object
        """
        visitor = MetricsVisitor()
        visitor.visit(tree)

        lines = source_code.split('\n')
        loc = len([line for line in lines if line.strip() and not line.strip().startswith('#')])

        # Calculate docstring coverage
        total_definitions = visitor.function_count + visitor.class_count
        documented = visitor.documented_count
        doc_coverage = documented / total_definitions if total_definitions > 0 else 0

        # Calculate comment ratio
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        comment_ratio = comment_lines / len(lines) if lines else 0

        return CodeMetrics(
            lines_of_code=loc,
            complexity=visitor.total_complexity,
            function_count=visitor.function_count,
            class_count=visitor.class_count,
            import_count=visitor.import_count,
            docstring_coverage=doc_coverage,
            comment_ratio=comment_ratio,
            max_function_complexity=visitor.max_complexity,
            duplicate_code_blocks=0,  # TODO: Implement duplicate detection
            unused_imports=visitor.unused_imports,
            unused_variables=visitor.unused_variables
        )

    async def _find_opportunities(self, tree: ast.AST, source_code: str, module_path: Path) -> List[Dict]:
        """
        Find improvement opportunities in the code.

        Args:
            tree: AST of the module
            source_code: Source code string
            module_path: Path to the module

        Returns:
            List of improvement opportunities
        """
        opportunities = []
        visitor = OpportunityFinder()
        visitor.visit(tree)

        # Performance opportunities
        for func in visitor.long_functions:
            opportunities.append({
                'type': 'refactor',
                'description': f"Function '{func}' is too long and should be refactored",
                'location': func,
                'priority': 'medium'
            })

        for func in visitor.complex_functions:
            opportunities.append({
                'type': 'optimization',
                'description': f"Function '{func}' has high complexity and could be simplified",
                'location': func,
                'priority': 'high'
            })

        # Missing error handling
        for func in visitor.missing_error_handling:
            opportunities.append({
                'type': 'reliability',
                'description': f"Function '{func}' lacks error handling",
                'location': func,
                'priority': 'high'
            })

        # Missing docstrings
        for item in visitor.missing_docstrings:
            opportunities.append({
                'type': 'documentation',
                'description': f"'{item}' is missing a docstring",
                'location': item,
                'priority': 'low'
            })

        return opportunities

    def _extract_dependencies(self, tree: ast.AST) -> Dict[str, List[str]]:
        """
        Extract module dependencies.

        Args:
            tree: AST of the module

        Returns:
            Dictionary of dependencies
        """
        dependencies = {
            'imports': [],
            'from_imports': [],
            'local_imports': [],
            'external_imports': []
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies['imports'].append(alias.name)
                    if alias.name.startswith('modules.'):
                        dependencies['local_imports'].append(alias.name)
                    else:
                        dependencies['external_imports'].append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                dependencies['from_imports'].append(module)
                if module.startswith('modules.'):
                    dependencies['local_imports'].append(module)
                else:
                    dependencies['external_imports'].append(module)

        return dependencies

    async def _ai_analysis(self, source_code: str, module_name: str) -> List[Dict]:
        """
        Use AI to find deeper improvement opportunities.

        Args:
            source_code: The source code to analyze
            module_name: Name of the module

        Returns:
            List of AI-discovered opportunities
        """
        if not self.llm_client:
            return []

        opportunities = []

        try:
            prompt = f"""Analyze this Python module '{module_name}' for improvement opportunities.
Focus on:
1. Performance optimizations
2. Code quality improvements
3. Potential bugs or issues
4. Architectural improvements

Code:
```python
{source_code[:3000]}  # Limit to first 3000 chars
```

Provide specific, actionable improvements. Format as JSON list with: type, description, priority.
Output ONLY the JSON list, nothing else."""

            response = self.llm_client.generate(
                model='qwen2.5:14b-instruct',
                prompt=prompt,
                options={'temperature': 0.3}
            )

            # Parse AI response
            import json
            try:
                ai_opportunities = json.loads(response['response'])
                for opp in ai_opportunities:
                    if isinstance(opp, dict) and 'type' in opp and 'description' in opp:
                        opportunities.append({
                            'type': opp.get('type', 'improvement'),
                            'description': opp.get('description'),
                            'location': module_name,
                            'priority': opp.get('priority', 'medium'),
                            'source': 'ai_analysis'
                        })
            except json.JSONDecodeError:
                self.logger.error("Failed to parse AI analysis response")

        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")

        return opportunities

    async def build_dependency_graph(self) -> Dict[str, Any]:
        """
        Build a dependency graph of the entire codebase.

        Returns:
            Dependency graph data
        """
        self.dependency_graph.clear()

        modules_path = self.phoenix_root / 'modules'
        for module_file in modules_path.glob('**/*.py'):
            if '__pycache__' in str(module_file):
                continue

            module_name = str(module_file.relative_to(self.phoenix_root).with_suffix(''))
            self.dependency_graph.add_node(module_name)

            with open(module_file, 'r') as f:
                try:
                    tree = ast.parse(f.read())
                    deps = self._extract_dependencies(tree)

                    for dep in deps['local_imports']:
                        dep_name = dep.replace('.', '/').replace('modules/', '')
                        self.dependency_graph.add_edge(module_name, dep_name)

                except SyntaxError:
                    continue

        # Find circular dependencies
        cycles = list(nx.simple_cycles(self.dependency_graph))

        return {
            'total_modules': self.dependency_graph.number_of_nodes(),
            'total_dependencies': self.dependency_graph.number_of_edges(),
            'circular_dependencies': cycles,
            'most_depended_on': self._find_most_depended_on(),
            'isolated_modules': list(nx.isolates(self.dependency_graph))
        }

    def _find_most_depended_on(self) -> List[Tuple[str, int]]:
        """Find the most depended-on modules."""
        in_degrees = dict(self.dependency_graph.in_degree())
        sorted_modules = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
        return sorted_modules[:5]


class MetricsVisitor(ast.NodeVisitor):
    """AST visitor for collecting code metrics."""

    def __init__(self):
        self.function_count = 0
        self.class_count = 0
        self.import_count = 0
        self.documented_count = 0
        self.total_complexity = 0
        self.max_complexity = 0
        self.current_complexity = 0
        self.unused_imports = []
        self.unused_variables = []
        self.used_names = set()

    def visit_FunctionDef(self, node):
        self.function_count += 1
        if ast.get_docstring(node):
            self.documented_count += 1

        # Calculate cyclomatic complexity
        prev_complexity = self.current_complexity
        self.current_complexity = 1  # Base complexity

        self.generic_visit(node)

        if self.current_complexity > self.max_complexity:
            self.max_complexity = self.current_complexity

        self.total_complexity += self.current_complexity
        self.current_complexity = prev_complexity

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)  # Treat async functions the same

    def visit_ClassDef(self, node):
        self.class_count += 1
        if ast.get_docstring(node):
            self.documented_count += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.current_complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.current_complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.current_complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.current_complexity += 1
        self.generic_visit(node)

    def visit_Import(self, node):
        self.import_count += 1
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.import_count += 1
        self.generic_visit(node)

    def visit_Name(self, node):
        self.used_names.add(node.id)
        self.generic_visit(node)


class OpportunityFinder(ast.NodeVisitor):
    """AST visitor for finding improvement opportunities."""

    def __init__(self):
        self.long_functions = []
        self.complex_functions = []
        self.missing_error_handling = []
        self.missing_docstrings = []
        self.current_function = None

    def visit_FunctionDef(self, node):
        self.current_function = node.name

        # Check function length
        if len(node.body) > 50:
            self.long_functions.append(node.name)

        # Check for docstring
        if not ast.get_docstring(node):
            self.missing_docstrings.append(node.name)

        # Check complexity (simplified)
        complexity = self._calculate_complexity(node)
        if complexity > 10:
            self.complex_functions.append(node.name)

        # Check for error handling
        has_try = any(isinstance(child, ast.Try) for child in ast.walk(node))
        if not has_try and 'init' not in node.name:
            self.missing_error_handling.append(node.name)

        self.generic_visit(node)
        self.current_function = None

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        if not ast.get_docstring(node):
            self.missing_docstrings.append(node.name)
        self.generic_visit(node)

    def _calculate_complexity(self, node):
        """Calculate cyclomatic complexity of a function."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        return complexity