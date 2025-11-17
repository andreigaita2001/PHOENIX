#!/usr/bin/env python3
"""
Autonomous Module Creator - Enables PHOENIX to create its own modules.
This is the key to true self-improvement and trustworthiness.
"""

import os
import ast
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import subprocess


class AutonomousModuleCreator:
    """
    Enables PHOENIX to autonomously create new modules based on needs.
    """

    def __init__(self, llm_client=None, memory_manager=None, safety_guard=None):
        """
        Initialize the Autonomous Module Creator.

        Args:
            llm_client: LLM for intelligent code generation
            memory_manager: Memory system for learning
            safety_guard: Safety checks before creating modules
        """
        self.llm_client = llm_client
        self.memory_manager = memory_manager
        self.safety_guard = safety_guard
        self.logger = logging.getLogger("PHOENIX.ModuleCreator")

        # Module templates
        self.templates = self._load_templates()

        # Track created modules
        self.created_modules = []

        # Phoenix root directory
        self.phoenix_root = Path(__file__).parent.parent

    def analyze_need(self, user_request: str, system_context: Dict) -> Dict[str, Any]:
        """
        Analyze if a new module is needed based on user request.

        Args:
            user_request: What the user is asking for
            system_context: Current system capabilities

        Returns:
            Analysis of whether a new module is needed
        """
        # Check existing capabilities
        existing_modules = self._get_existing_modules()

        analysis = {
            'need_detected': False,
            'module_type': None,
            'requirements': [],
            'confidence': 0.0,
            'reasoning': ''
        }

        # Use LLM to analyze the request
        if self.llm_client:
            prompt = f"""
Analyze this user request to determine if a new module is needed:

User Request: {user_request}

Existing Modules: {', '.join(existing_modules)}

Current Context: {json.dumps(system_context, indent=2)}

Determine:
1. Is a new module needed? (yes/no)
2. What type of module? (scheduler, analyzer, interface, tool, etc.)
3. What are the key requirements?
4. How confident are you? (0-1)
5. Explain your reasoning

Format as JSON.
"""

            try:
                response = self.llm_client.chat(
                    model='qwen2.5:14b-instruct',
                    messages=[
                        {'role': 'system', 'content': 'You are an expert system architect.'},
                        {'role': 'user', 'content': prompt}
                    ]
                )

                # Parse response
                result = json.loads(response['message']['content'])
                analysis.update(result)

            except Exception as e:
                self.logger.error(f"Failed to analyze need: {e}")

        return analysis

    def design_module(self, module_type: str, requirements: List[str]) -> Dict[str, Any]:
        """
        Design a new module based on requirements.

        Args:
            module_type: Type of module to create
            requirements: List of requirements

        Returns:
            Module design specification
        """
        design = {
            'name': '',
            'description': '',
            'class_name': '',
            'methods': [],
            'dependencies': [],
            'integration_points': [],
            'test_cases': []
        }

        if self.llm_client:
            prompt = f"""
Design a Python module for PHOENIX AI system:

Module Type: {module_type}
Requirements:
{chr(10).join(f'- {req}' for req in requirements)}

The module should:
1. Follow PHOENIX's architecture patterns
2. Integrate with existing memory and safety systems
3. Be reliable and trustworthy
4. Include proper error handling
5. Have clear documentation

Provide:
- Module name and description
- Main class name
- Key methods with docstrings
- Required dependencies
- Integration points with other PHOENIX modules
- Test cases

Format as JSON.
"""

            try:
                response = self.llm_client.chat(
                    model='qwen2.5:14b-instruct',
                    messages=[
                        {'role': 'system', 'content': 'You are an expert Python developer creating modules for an AI system.'},
                        {'role': 'user', 'content': prompt}
                    ]
                )

                design = json.loads(response['message']['content'])

            except Exception as e:
                self.logger.error(f"Failed to design module: {e}")

        return design

    def generate_module_code(self, design: Dict[str, Any]) -> str:
        """
        Generate actual Python code for the module.

        Args:
            design: Module design specification

        Returns:
            Generated Python code
        """
        # Use template or generate from scratch
        if design.get('module_type') in self.templates:
            code = self.templates[design['module_type']].format(**design)
        else:
            code = self._generate_from_design(design)

        # Validate the code
        if not self._validate_code(code):
            self.logger.error("Generated code failed validation")
            return None

        return code

    def _generate_from_design(self, design: Dict[str, Any]) -> str:
        """Generate code from design specification."""
        if not self.llm_client:
            return self._generate_basic_template(design)

        prompt = f"""
Generate complete Python code for this module design:

{json.dumps(design, indent=2)}

Requirements:
1. Include proper imports
2. Add comprehensive docstrings
3. Implement all specified methods
4. Add error handling
5. Include logging
6. Make it production-ready

Generate the complete .py file content.
"""

        try:
            response = self.llm_client.chat(
                model='qwen2.5:14b-instruct',
                messages=[
                    {'role': 'system', 'content': 'You are an expert Python developer. Generate clean, reliable code.'},
                    {'role': 'user', 'content': prompt}
                ]
            )

            return response['message']['content']

        except Exception as e:
            self.logger.error(f"Failed to generate code: {e}")
            return self._generate_basic_template(design)

    def _generate_basic_template(self, design: Dict[str, Any]) -> str:
        """Generate a basic module template."""
        return f'''#!/usr/bin/env python3
"""
{design.get('name', 'New Module')} - {design.get('description', 'Auto-generated module')}
Generated by PHOENIX Autonomous Module Creator
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime


class {design.get('class_name', 'NewModule')}:
    """
    {design.get('description', 'Auto-generated module for PHOENIX')}
    """

    def __init__(self, config: Dict = None, memory_manager=None):
        """
        Initialize the module.

        Args:
            config: Configuration dictionary
            memory_manager: PHOENIX memory system
        """
        self.config = config or {{}}
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(f"PHOENIX.{design.get('class_name', 'NewModule')}")

        self.logger.info(f"{design.get('name', 'Module')} initialized")

    def process(self, input_data: Any) -> Any:
        """
        Main processing method.

        Args:
            input_data: Input to process

        Returns:
            Processed result
        """
        try:
            # Implementation goes here
            result = self._process_internal(input_data)

            # Store in memory if available
            if self.memory_manager:
                self.memory_manager.learn_fact(
                    f"Processed: {{input_data}}",
                    category=self.__class__.__name__
                )

            return result

        except Exception as e:
            self.logger.error(f"Processing failed: {{e}}")
            raise

    def _process_internal(self, data: Any) -> Any:
        """Internal processing logic."""
        # TODO: Implement actual processing
        return data

    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {{
            'name': '{design.get('name', 'Module')}',
            'status': 'active',
            'created': datetime.now().isoformat()
        }}
'''

    def _validate_code(self, code: str) -> bool:
        """Validate generated Python code."""
        try:
            # Parse the code to check syntax
            ast.parse(code)

            # Check for dangerous operations
            if self.safety_guard:
                validation = self.safety_guard.validate_code(code)
                if not validation['safe']:
                    self.logger.error(f"Code failed safety check: {validation['reason']}")
                    return False

            return True

        except SyntaxError as e:
            self.logger.error(f"Syntax error in generated code: {e}")
            return False

    def create_and_integrate(self, design: Dict[str, Any], code: str) -> bool:
        """
        Create the module file and integrate it into PHOENIX.

        Args:
            design: Module design
            code: Generated code

        Returns:
            Success status
        """
        try:
            # Determine module path
            module_name = design['name'].lower().replace(' ', '_')
            module_path = self.phoenix_root / 'modules' / f'{module_name}.py'

            # Check if module already exists
            if module_path.exists():
                self.logger.warning(f"Module {module_name} already exists")
                return False

            # Write the module
            with open(module_path, 'w') as f:
                f.write(code)

            self.logger.info(f"Created module: {module_path}")

            # Update __init__.py if needed
            self._update_init_file(module_name, design['class_name'])

            # Run tests if available
            if design.get('test_cases'):
                self._run_module_tests(module_name, design['test_cases'])

            # Track creation
            self.created_modules.append({
                'name': module_name,
                'path': str(module_path),
                'created': datetime.now().isoformat(),
                'design': design
            })

            # Store in memory
            if self.memory_manager:
                self.memory_manager.learn_fact(
                    f"Created module: {module_name} - {design['description']}",
                    category='self_modification'
                )

            return True

        except Exception as e:
            self.logger.error(f"Failed to create module: {e}")
            return False

    def _update_init_file(self, module_name: str, class_name: str):
        """Update the modules __init__.py file."""
        init_file = self.phoenix_root / 'modules' / '__init__.py'

        if init_file.exists():
            with open(init_file, 'a') as f:
                f.write(f"\n# Auto-imported by Module Creator\n")
                f.write(f"try:\n")
                f.write(f"    from .{module_name} import {class_name}\n")
                f.write(f"except ImportError:\n")
                f.write(f"    {class_name} = None\n")

    def _run_module_tests(self, module_name: str, test_cases: List[Dict]):
        """Run tests on the newly created module."""
        # TODO: Implement test execution
        self.logger.info(f"Would run {len(test_cases)} tests for {module_name}")

    def _get_existing_modules(self) -> List[str]:
        """Get list of existing PHOENIX modules."""
        modules_dir = self.phoenix_root / 'modules'

        if not modules_dir.exists():
            return []

        modules = []
        for file in modules_dir.glob('*.py'):
            if file.name != '__init__.py':
                modules.append(file.stem)

        return modules

    def _load_templates(self) -> Dict[str, str]:
        """Load module templates."""
        return {
            'scheduler': self._get_scheduler_template(),
            'analyzer': self._get_analyzer_template(),
            'interface': self._get_interface_template()
        }

    def _get_scheduler_template(self) -> str:
        """Get scheduler module template."""
        return '''#!/usr/bin/env python3
"""
{name} - {description}
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json


class {class_name}:
    """Scheduler module for managing tasks and events."""

    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager
        self.logger = logging.getLogger("PHOENIX.{class_name}")
        self.tasks = []
        self.running = False

    async def schedule_task(self, task: Dict[str, Any]) -> str:
        """Schedule a new task."""
        task_id = self._generate_task_id()
        task['id'] = task_id
        task['created'] = datetime.now().isoformat()
        task['status'] = 'scheduled'

        self.tasks.append(task)

        if self.memory_manager:
            self.memory_manager.learn_fact(
                f"Scheduled task: {task.get('name', 'Unknown')}",
                category='scheduling'
            )

        return task_id

    async def run(self):
        """Run the scheduler."""
        self.running = True
        while self.running:
            await self._process_tasks()
            await asyncio.sleep(60)  # Check every minute

    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        import hashlib
        return hashlib.md5(f"{datetime.now()}".encode()).hexdigest()[:8]

    def _process_tasks(self):
        """Process scheduled tasks."""
        now = datetime.now()
        for task in self.tasks:
            if task['status'] == 'scheduled':
                # Check if it's time to run
                scheduled_time = datetime.fromisoformat(task['scheduled_for'])
                if now >= scheduled_time:
                    self._execute_task(task)

    def _execute_task(self, task: Dict):
        """Execute a scheduled task."""
        self.logger.info(f"Executing task: {task.get('name')}")
        task['status'] = 'completed'
        task['completed_at'] = datetime.now().isoformat()
'''

    def _get_analyzer_template(self) -> str:
        """Get analyzer module template."""
        return '''#!/usr/bin/env python3
"""
{name} - {description}
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime


class {class_name}:
    """Analyzer module for data analysis and insights."""

    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager
        self.logger = logging.getLogger("PHOENIX.{class_name}")
        self.analysis_results = []

    def analyze(self, data: Any) -> Dict[str, Any]:
        """Analyze provided data."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'data_type': type(data).__name__,
            'insights': []
        }

        # Perform analysis
        insights = self._perform_analysis(data)
        result['insights'] = insights

        # Store results
        self.analysis_results.append(result)

        if self.memory_manager:
            self.memory_manager.learn_fact(
                f"Analysis completed: {len(insights)} insights found",
                category='analysis'
            )

        return result

    def _perform_analysis(self, data: Any) -> List[Dict]:
        """Perform actual analysis."""
        insights = []
        # TODO: Implement actual analysis logic
        return insights
'''

    def _get_interface_template(self) -> str:
        """Get interface module template."""
        return '''#!/usr/bin/env python3
"""
{name} - {description}
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime


class {class_name}:
    """Interface module for external interactions."""

    def __init__(self, config: Dict = None, memory_manager=None):
        self.config = config or {}
        self.memory_manager = memory_manager
        self.logger = logging.getLogger("PHOENIX.{class_name}")
        self.connections = []

    def connect(self, target: str, params: Dict = None) -> bool:
        """Connect to external service or system."""
        try:
            connection = {
                'target': target,
                'params': params or {},
                'established': datetime.now().isoformat(),
                'status': 'connected'
            }

            self.connections.append(connection)

            if self.memory_manager:
                self.memory_manager.learn_fact(
                    f"Connected to: {target}",
                    category='connections'
                )

            return True

        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False

    def send(self, data: Any, target: str = None) -> bool:
        """Send data through interface."""
        # TODO: Implement sending logic
        return True

    def receive(self) -> Any:
        """Receive data from interface."""
        # TODO: Implement receiving logic
        return None
'''