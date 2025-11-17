#!/usr/bin/env python3
"""
PHOENIX Core - The main AI engine
This is the brain of your AI system. It coordinates all other components.
"""

import os
import sys
import yaml
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add parent directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Import modules that exist
try:
    from modules.system_control import SystemControl
except ImportError:
    SystemControl = None

try:
    from modules.memory_manager import MemoryManager
except ImportError:
    MemoryManager = None

try:
    from modules.command_parser import CommandParser
except ImportError:
    CommandParser = None

try:
    from modules.safety_guard import SafetyGuard
except ImportError:
    SafetyGuard = None

try:
    from modules.advanced_learning import AdvancedLearning
except ImportError:
    AdvancedLearning = None

try:
    from modules.automation import AutomationEngine
except ImportError:
    AutomationEngine = None

try:
    from modules.intelligent_executor import IntelligentExecutor
except ImportError:
    IntelligentExecutor = None

try:
    from modules.system_discovery import SystemDiscovery
except ImportError:
    SystemDiscovery = None

try:
    from modules.self_awareness import SelfAwareness
except ImportError:
    SelfAwareness = None

# New modules - Multi-Model Intelligence
try:
    from modules.multi_model.multi_model_coordinator import MultiModelCoordinator
except ImportError:
    MultiModelCoordinator = None

# New modules - Active Learning
try:
    from modules.active_learning.system_scanner import SystemScanner
    from modules.active_learning.pattern_recognition import PatternRecognitionEngine
    from modules.active_learning.habit_learning import HabitLearner
    from modules.active_learning.predictive_modeling import PredictiveModel
    from modules.active_learning.knowledge_consolidation import KnowledgeConsolidator
except ImportError:
    SystemScanner = None
    PatternRecognitionEngine = None
    HabitLearner = None
    PredictiveModel = None
    KnowledgeConsolidator = None

# Self-Modification Framework
try:
    from modules.self_modification.modification_engine import ModificationEngine
    from modules.autonomous_module_creator import AutonomousModuleCreator
except ImportError:
    ModificationEngine = None
    AutonomousModuleCreator = None

# New Essential Modules for Non-Hallucination
try:
    from modules.capability_manager import CapabilityManager
    from modules.web_search import WebSearchModule
    from modules.tennis_scheduler import TennisScheduler
    from modules.gui_manager import GUIManager
    from modules.code_executor import CodeExecutor
except ImportError as e:
    print(f"Warning: Some modules failed to import: {e}")
    CapabilityManager = None
    WebSearchModule = None
    TennisScheduler = None
    GUIManager = None
    CodeExecutor = None

# Personal Data and Knowledge Modules
try:
    from modules.personal_data_vault import PersonalDataVault
    from modules.personal_knowledge_extractor import PersonalKnowledgeExtractor
except ImportError as e:
    print(f"Warning: Personal data modules failed to import: {e}")
    PersonalDataVault = None
    PersonalKnowledgeExtractor = None

# Web Browser Module
try:
    from modules.web_browser import WebBrowser
except ImportError as e:
    print(f"Warning: Web browser module failed to import: {e}")
    WebBrowser = None

# Research Agent Module
try:
    from modules.research_agent import ResearchAgent
except ImportError as e:
    print(f"Warning: Research agent module failed to import: {e}")
    ResearchAgent = None

class PhoenixCore:
    """
    The main AI system class that coordinates all components.
    Think of this as the 'consciousness' of your AI.
    """

    def __init__(self, config_path: str = "config/phoenix_config.yaml"):
        """
        Initialize Phoenix with configuration.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.name = "PHOENIX"
        self.version = "0.1.0"
        self.start_time = datetime.now()

        # Load configuration
        self.config = self._load_config(config_path)

        # Set up logging
        self._setup_logging()

        # Initialize state
        self.is_running = False
        self.modules = {}
        self.current_task = None
        self.command_parser = CommandParser() if CommandParser else None
        self.intelligent_executor = None  # Will be initialized after LLM

        self.logger.info(f"{self.name} v{self.version} initializing...")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        config_file = Path(config_path)
        if not config_file.exists():
            # Use default config if file doesn't exist
            return self._get_default_config()

        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration if no config file exists.

        Returns:
            Default configuration dictionary
        """
        return {
            'system': {
                'name': 'PHOENIX',
                'version': '0.1.0',
                'mode': 'development'
            },
            'llm': {
                'provider': 'ollama',
                'model': 'qwen2.5:14b',
                'temperature': 0.7
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/phoenix.log'
            }
        }

    def _setup_logging(self):
        """
        Set up logging system for tracking all activities.
        This is crucial for debugging and understanding what your AI is doing.
        """
        # Create logs directory if it doesn't exist
        log_dir = Path(self.config['logging']['file']).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['logging']['file']),
                logging.StreamHandler()  # Also output to console
            ]
        )

        self.logger = logging.getLogger(self.name)

    async def initialize_modules(self):
        """
        Initialize all system modules.
        Each module handles a specific capability of the AI.
        """
        self.logger.info("Initializing modules...")

        try:
            # We'll uncomment these as we create each module

            # Initialize LLM connection
            self.logger.info("Connecting to Ollama...")
            await self._init_llm()

            # Initialize Intelligent Executor (needs LLM)
            if IntelligentExecutor and hasattr(self, 'llm_client'):
                self.logger.info("Initializing Intelligent Executor...")
                self.intelligent_executor = IntelligentExecutor(self.llm_client)

            # Initialize memory system
            if MemoryManager:
                self.logger.info("Initializing memory systems...")
                self.modules['memory'] = MemoryManager(self.config.get('memory', {}))

            # Initialize safety guard
            if SafetyGuard:
                self.logger.info("Initializing safety mechanisms...")
                self.modules['safety'] = SafetyGuard(self.config.get('safety', {}))

            # Initialize system control
            if SystemControl:
                self.logger.info("Initializing system control...")
                self.modules['system'] = SystemControl(self.config.get('system_control', {}))

            # Initialize advanced learning
            if AdvancedLearning:
                self.logger.info("Initializing advanced learning...")
                self.modules['learning'] = AdvancedLearning(self.config.get('learning', {}))

            # Initialize automation engine
            if AutomationEngine:
                self.logger.info("Initializing automation engine...")
                self.modules['automation'] = AutomationEngine(
                    self.config.get('automation', {}),
                    system_control=self.modules.get('system'),
                    memory_manager=self.modules.get('memory')
                )
                # Start automation
                await self.modules['automation'].start()

            # Initialize system discovery
            if SystemDiscovery:
                self.logger.info("Initializing system discovery...")
                self.modules['discovery'] = SystemDiscovery(
                    system_control=self.modules.get('system'),
                    memory_manager=self.modules.get('memory'),
                    learning=self.modules.get('learning')
                )

            # Initialize self-awareness
            if SelfAwareness:
                self.logger.info("Initializing self-awareness...")
                self.modules['self_awareness'] = SelfAwareness(
                    system_control=self.modules.get('system'),
                    memory=self.modules.get('memory')
                )

            # Initialize Multi-Model Intelligence System
            if MultiModelCoordinator:
                self.logger.info("Initializing Multi-Model Intelligence...")
                self.modules['multi_model'] = MultiModelCoordinator(
                    config=self.config.get('multi_model', {}),
                    memory_manager=self.modules.get('memory')
                )
                # Initialize the multi-model system WITHOUT auto-benchmarking
                init_config = self.config.get('multi_model', {})
                init_config['auto_benchmark'] = False  # Don't benchmark on startup
                self.modules['multi_model'].config = init_config
                await self.modules['multi_model'].initialize()

            # Initialize Active Learning System
            if SystemScanner and PatternRecognitionEngine:
                self.logger.info("Initializing Active Learning System...")

                # System Scanner
                self.modules['scanner'] = SystemScanner(
                    memory_manager=self.modules.get('memory'),
                    multi_model=self.modules.get('multi_model')
                )

                # Pattern Recognition
                self.modules['pattern_engine'] = PatternRecognitionEngine(
                    memory_manager=self.modules.get('memory')
                )
                # Load existing patterns if they exist
                pattern_file = Path('./data/patterns.json')
                if pattern_file.exists():
                    self.modules['pattern_engine'].import_patterns(pattern_file)
                    self.logger.info(f"Loaded existing patterns from {pattern_file}")

                # Habit Learning
                if HabitLearner:
                    self.modules['habit_learner'] = HabitLearner(
                        pattern_engine=self.modules['pattern_engine'],
                        memory_manager=self.modules.get('memory')
                    )
                    # Load existing habits if they exist
                    habit_file = Path('./data/habits.json')
                    if habit_file.exists():
                        self.modules['habit_learner'].import_habits(habit_file)
                        self.logger.info(f"Loaded existing habits from {habit_file}")

                # Predictive Modeling
                if PredictiveModel:
                    self.modules['predictive_model'] = PredictiveModel(
                        pattern_engine=self.modules['pattern_engine'],
                        habit_learner=self.modules.get('habit_learner'),
                        memory_manager=self.modules.get('memory')
                    )

                # Knowledge Consolidation
                if KnowledgeConsolidator:
                    self.modules['knowledge'] = KnowledgeConsolidator(
                        scanner=self.modules['scanner'],
                        pattern_engine=self.modules['pattern_engine'],
                        habit_learner=self.modules.get('habit_learner'),
                        predictive_model=self.modules.get('predictive_model'),
                        memory_manager=self.modules.get('memory')
                    )

                # Don't do full scan on startup - too slow
                # User can run 'familiarize' command when ready
                self.logger.info("System scanner ready. Run 'familiarize' for full scan.")

            # Initialize Self-Modification Framework
            if ModificationEngine and hasattr(self, 'llm_client'):
                self.logger.info("Initializing Self-Modification Framework...")
                self.modules['modification'] = ModificationEngine(
                    llm_client=self.llm_client,
                    memory=self.modules.get('memory')
                )

            # Initialize Autonomous Module Creator
            if AutonomousModuleCreator and hasattr(self, 'llm_client'):
                self.logger.info("Initializing Autonomous Module Creator...")
                self.modules['module_creator'] = AutonomousModuleCreator(
                    llm_client=self.llm_client,
                    memory_manager=self.modules.get('memory'),
                    safety_guard=self.modules.get('safety')
                )

            # Initialize Web Search Module
            if WebSearchModule:
                self.logger.info("Initializing Web Search Module...")
                self.modules['web_search'] = WebSearchModule(
                    memory_manager=self.modules.get('memory')
                )

            # Initialize Tennis Scheduler
            if TennisScheduler:
                self.logger.info("Initializing Tennis Scheduler...")
                self.modules['tennis_scheduler'] = TennisScheduler(
                    memory_manager=self.modules.get('memory')
                )

            # Initialize GUI Manager
            if GUIManager:
                self.logger.info("Initializing GUI Manager...")
                self.modules['gui_manager'] = GUIManager(
                    system_control=self.modules.get('system')
                )

            # Initialize Code Executor
            if CodeExecutor:
                self.logger.info("Initializing Code Executor...")
                self.modules['code_executor'] = CodeExecutor(
                    system_control=self.modules.get('system')
                )

            # Initialize Web Browser
            if WebBrowser:
                self.logger.info("Initializing Web Browser...")
                self.modules['web_browser'] = WebBrowser(
                    memory_manager=self.modules.get('memory')
                )
                self.logger.info("Web Browser ready - Full web access enabled")

            # Initialize Research Agent
            if ResearchAgent and self.modules.get('web_browser'):
                self.logger.info("Initializing Research Agent...")
                self.modules['research_agent'] = ResearchAgent(
                    web_browser=self.modules['web_browser'],
                    memory_manager=self.modules.get('memory'),
                    llm_client=self.llm_client if hasattr(self, 'llm_client') else None
                )
                self.logger.info("Research Agent ready - Autonomous research enabled")

            # Initialize Personal Data Vault
            if PersonalDataVault:
                self.logger.info("Initializing Personal Data Vault...")
                self.modules['personal_vault'] = PersonalDataVault(
                    memory_manager=self.modules.get('memory')
                )
                self.logger.info("Personal Data Vault ready - Maximum privacy enabled")

            # Initialize Personal Knowledge Extractor
            if PersonalKnowledgeExtractor and self.modules.get('personal_vault'):
                self.logger.info("Initializing Personal Knowledge Extractor...")
                self.modules['knowledge_extractor'] = PersonalKnowledgeExtractor(
                    vault=self.modules['personal_vault'],
                    memory_manager=self.modules.get('memory')
                )
                self.logger.info("Knowledge Extractor ready - Pattern analysis enabled")

            # Initialize Capability Manager (MUST be last to scan all modules)
            if CapabilityManager:
                self.logger.info("Initializing Capability Manager...")
                self.modules['capability_manager'] = CapabilityManager(
                    modules=self.modules,
                    module_creator=self.modules.get('module_creator')
                )
                # Generate initial capability report
                self.logger.info("Capability scan complete")
                self.capability_report = self.modules['capability_manager'].generate_capability_report()

            self.logger.info("All modules initialized successfully!")

        except Exception as e:
            self.logger.error(f"Failed to initialize modules: {e}")
            raise

    async def _init_llm(self):
        """
        Initialize connection to the Local Language Model (Ollama).
        This is the 'thinking' part of your AI.
        """
        try:
            import ollama

            # Check if Ollama is running
            self.llm_client = ollama.Client()

            # Test the connection with proper error handling
            try:
                models = self.llm_client.list()
                # The response is a dictionary with 'models' key
                model_list = models.get('models', [])
            except:
                # Fallback - assume model exists if ollama is running
                model_list = []
                self.logger.warning("Could not list models, assuming configured model exists")
            model_names = []
            for m in model_list:
                # Each model dict has a 'name' key
                if isinstance(m, dict) and 'name' in m:
                    model_names.append(m['name'])

            self.logger.info(f"Connected to Ollama. Available models: {model_names}")

            # Check if our preferred model is available
            model_name = self.config['llm']['model']
            # Only warn if we successfully got model list AND model isn't found
            if model_names and not any(model_name in name for name in model_names):
                self.logger.warning(f"Model {model_name} not in list, but may exist")
                # Don't auto-pull - often the model exists but list() failed

        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            self.logger.info("Please make sure Ollama is installed and running.")
            self.logger.info("Install: curl -fsSL https://ollama.com/install.sh | sh")
            self.logger.info("Run: ollama serve")
            raise

    async def think(self, prompt: str, context: Optional[Dict] = None) -> str:
        """
        Process a prompt through the LLM to generate a response.
        This is how the AI 'thinks' about problems.

        Args:
            prompt: The question or task to think about
            context: Additional context for the LLM

        Returns:
            The AI's response
        """
        try:
            import ollama

            # Build conversation messages with history
            messages = []

            # Add system message defining PHOENIX's identity
            system_message = """You are PHOENIX, an advanced AI system with these ACTUAL capabilities:
- Self-modification and improvement
- Multi-model intelligence (can switch between specialized AI models)
- Active learning (constantly learning from user patterns)
- System control (can execute commands and manage files)
- Memory persistence (remember all conversations and learn from them)
- Web search (can search the web for information)
- Tennis scheduling (can manage tennis lesson schedules)
- GUI creation (can create graphical interfaces)

IMPORTANT: You CANNOT:
- Send emails (no email capability yet)
- Control web browsers directly
- Send real-time notifications
- Make direct API calls to external services
- Voice interaction

ALWAYS be honest about what you can and cannot do. If asked to do something you cannot do, say:
"I don't have that capability yet, but I could create a module for it if needed."

You are running locally on the user's computer and have direct system access.
You should be helpful, intelligent, and proactive in learning about the user's needs."""

            # Add conversation history if available
            if 'memory' in self.modules:
                # Get recent conversation history
                recent_convos = self.modules['memory'].get_recent_memories(10)
                if recent_convos:
                    for convo in recent_convos:
                        if 'user' in convo:
                            messages.append({'role': 'user', 'content': convo['user']})
                        if 'assistant' in convo:
                            messages.append({'role': 'assistant', 'content': convo['assistant']})

                # Get similar past conversations for context
                similar_memories = self.modules['memory'].recall_similar(prompt, limit=3)
                if similar_memories:
                    context_info = "Relevant past interactions:\n"
                    for mem in similar_memories:
                        context_info += f"- {mem['content'][:100]}...\n"
                    system_message += f"\n\n{context_info}"

                # Add user preferences
                user_info = self.modules['memory'].get_user_info()
                if user_info:
                    if user_info.get('name'):
                        system_message += f"\n\nUser's name is {user_info['name']}."
                    if user_info.get('preferences'):
                        system_message += f"\nUser preferences: {user_info['preferences']}"

            # Add the system message at the beginning
            messages.insert(0, {'role': 'system', 'content': system_message})

            # Add the current user prompt
            messages.append({'role': 'user', 'content': prompt})

            # Get response from LLM using chat API for better context
            response = self.llm_client.chat(
                model=self.config['llm']['model'],
                messages=messages,
                options={
                    'temperature': self.config['llm'].get('temperature', 0.7),
                    'num_predict': self.config['llm'].get('max_tokens', 4096)
                }
            )

            return response['message']['content']

        except Exception as e:
            self.logger.error(f"Error during thinking: {e}")
            # Fallback to a simple response
            return f"I encountered an error while processing: {e}"

    async def process_command(self, command: str) -> str:
        """
        Process a user command and execute appropriate actions.
        This is the main interface for interacting with the AI.

        Args:
            command: The user's command or question

        Returns:
            The AI's response and action result
        """
        self.logger.info(f"Processing command: {command}")

        # Record command for pattern recognition
        if 'pattern_engine' in self.modules:
            self.modules['pattern_engine'].record_event('command', {
                'command': command,
                'directory': os.getcwd(),
                'timestamp': datetime.now().isoformat()
            })

        try:
            # First, check if this is about PHOENIX itself
            if 'self_awareness' in self.modules:
                understanding = self.modules['self_awareness'].understand_self_request(command)

                # Handle self-exploration requests
                if understanding.get('wants_exploration') and understanding.get('wants_understanding'):
                    self.logger.info("Self-exploration request detected")
                    result = self.modules['self_awareness'].execute_self_exploration()

                    # Store in memory
                    if 'memory' in self.modules:
                        self.modules['memory'].remember_conversation(command, result)

                    return result

            # NEW: Check capabilities before processing to prevent hallucination
            if 'capability_manager' in self.modules:
                # Check for specific action requests
                command_lower = command.lower()

                # Email request - catch early to prevent hallucination
                if any(phrase in command_lower for phrase in ['send email', 'send an email', 'email to', 'send a message']):
                    return "I don't have email capability yet, but I could create an email module if needed. Would you like me to do that?"

                # Personal data ingestion requests - CHECK BEFORE WEB SEARCH
                if any(phrase in command_lower for phrase in ['ingest google takeout', 'import my data', 'load my personal data', 'process takeout']):
                    if 'personal_vault' in self.modules:
                        # Extract path from command
                        import re
                        path_match = re.search(r'(?:from |at |in )"?([^"]+)"?$', command)

                        if path_match:
                            takeout_path = path_match.group(1).strip('"')
                        else:
                            return "Please specify the path to your Google Takeout folder. Example: 'ingest google takeout from /path/to/Takeout'"

                        self.logger.info(f"Starting Google Takeout ingestion from {takeout_path}")
                        result = self.modules['personal_vault'].ingest_google_takeout(takeout_path)

                        if result['success']:
                            response = f"✅ **Successfully ingested your personal data!**\n\n"
                            response += f"• Categories processed: {', '.join(result['categories_processed'])}\n"
                            response += f"• Total items: {result['total_items']:,}\n"
                            response += f"• Privacy: All data encrypted with AES-256\n"
                            response += f"• Storage: Local only (no cloud)\n\n"

                            # Run knowledge extraction
                            if 'knowledge_extractor' in self.modules:
                                self.logger.info("Analyzing personal patterns...")
                                analysis = self.modules['knowledge_extractor'].analyze_personal_data()
                                response += f"**Pattern Analysis:**\n"
                                response += f"• Patterns found: {analysis['patterns_found']}\n"
                                response += f"• Insights generated: {analysis['insights_generated']}\n"
                                response += f"• Interests detected: {analysis['interests_detected']}\n"

                            privacy_report = self.modules['personal_vault'].get_privacy_report()
                            response += f"\n**Privacy Status:** {privacy_report['privacy_level']}"

                            if 'memory' in self.modules:
                                self.modules['memory'].remember_conversation(command, response)
                            return response
                        else:
                            return f"❌ Failed to ingest data: {', '.join(result['errors'])}"
                    else:
                        return "Personal data vault is not initialized. Cannot ingest data."

                # Research requests - DEEP AUTONOMOUS RESEARCH
                if any(phrase in command_lower for phrase in ['research on', 'do research on', 'research about', 'thoroughly research', 'deep dive on', 'investigate']):
                    if 'research_agent' in self.modules:
                        # Extract research topic
                        import re
                        patterns = [
                            r'research (?:on |about )(.+)',
                            r'do research on (.+)',
                            r'thoroughly research (.+)',
                            r'deep dive on (.+)',
                            r'investigate (.+)'
                        ]

                        topic = None
                        for pattern in patterns:
                            match = re.search(pattern, command_lower)
                            if match:
                                topic = match.group(1).strip()
                                break

                        if topic:
                            self.logger.info(f"Starting deep research on: {topic}")

                            # Determine objective from command
                            objective = None
                            if 'to implement' in command_lower:
                                objective = f"Find the best way to implement {topic}"
                            elif 'best' in command_lower:
                                objective = f"Find the best solution for {topic}"
                            elif 'how to' in command_lower:
                                objective = f"Understand how to work with {topic}"

                            # Start research
                            response = "🔬 **Starting Deep Research**\n\n"
                            response += f"Topic: {topic}\n"
                            if objective:
                                response += f"Objective: {objective}\n\n"

                            response += "⏳ Conducting thorough research...\n"
                            response += "• Phase 1: Source Discovery\n"
                            response += "• Phase 2: Deep Dive\n"
                            response += "• Phase 3: Verification\n"
                            response += "• Phase 4: Synthesis\n\n"

                            # Perform research
                            research_results = self.modules['research_agent'].research(topic, objective)

                            # Format results
                            response += f"\n✅ **Research Complete!**\n\n"
                            response += f"**Sources Consulted:** {research_results['sources_consulted']}\n"
                            response += f"**Confidence Level:** {research_results['confidence']*100:.0f}%\n"
                            response += f"**Time Taken:** {research_results['time_taken']} seconds\n\n"

                            response += "**📊 Synthesis:**\n"
                            response += research_results['synthesis'][:1000] + "...\n\n"

                            if research_results['recommendations']:
                                response += "**💡 Recommendations:**\n"
                                for rec in research_results['recommendations'][:3]:
                                    response += f"• {rec['recommendation']}\n"
                                response += "\n"

                            if research_results['key_findings']:
                                response += "**🔍 Key Findings:**\n"
                                for finding in research_results['key_findings'][:5]:
                                    response += f"• {finding.get('title', 'Finding')}: {finding.get('key_points', [''])[0][:100]}...\n"

                            response += "\n**Research Trail:**\n"
                            for step in research_results['research_trail'][:5]:
                                response += f"• {step}\n"

                            # Store in memory
                            if 'memory' in self.modules:
                                self.modules['memory'].remember_conversation(command, response)
                                self.modules['memory'].learn_fact(
                                    f"Researched {topic}: {research_results['synthesis'][:200]}",
                                    'research'
                                )

                            return response
                        else:
                            return "Please specify what you want me to research. Example: 'research on best Python web frameworks'"
                    else:
                        return "Research Agent not initialized. Cannot conduct deep research."

                # Research feedback
                if 'research feedback' in command_lower or 'research was' in command_lower:
                    if 'research_agent' in self.modules:
                        # Parse feedback
                        feedback = {
                            'depth_rating': 5 if 'thorough' in command_lower or 'excellent' in command_lower else 3,
                            'source_quality': 5 if 'good sources' in command_lower else 3,
                            'too_long': 'too long' in command_lower or 'took forever' in command_lower,
                            'comments': command
                        }

                        result = self.modules['research_agent'].provide_feedback('latest', feedback)

                        response = "📝 **Feedback Received!**\n\n"
                        response += "Improvements applied:\n"
                        for improvement in result['improvements_applied']:
                            response += f"• {improvement['reason']}\n"

                        response += "\n**Updated Methodology:**\n"
                        for key, value in result['current_methodology'].items():
                            response += f"• {key}: {value}\n"

                        return response

                # Web browsing requests
                if any(phrase in command_lower for phrase in ['browse to', 'go to website', 'navigate to', 'visit website', 'open website']):
                    if 'web_browser' in self.modules:
                        # Extract URL
                        import re
                        url_match = re.search(r'(?:browse to |go to |navigate to |visit |open )(https?://[^\s]+|www\.[^\s]+|[^\s]+\.[^\s]+)', command_lower)
                        if url_match:
                            url = url_match.group(1)
                            if not url.startswith('http'):
                                url = 'https://' + url

                            self.logger.info(f"Navigating to: {url}")
                            result = self.modules['web_browser'].navigate(url)

                            if result['success']:
                                response = f"🌐 **Navigated to: {result['title']}**\n\n"
                                response += f"URL: {result['url']}\n\n"
                                response += f"**Page Content (excerpt):**\n{result['text'][:500]}...\n\n"
                                response += f"Found {len(result['links'])} links on the page."

                                if 'memory' in self.modules:
                                    self.modules['memory'].remember_conversation(command, response)
                                return response
                            else:
                                return f"❌ Failed to navigate: {result['error']}"
                    else:
                        return "Web browser module not initialized."

                # Web search requests - use the browser for real searching
                if any(phrase in command_lower for phrase in ['search the web for', 'search online for', 'look up online', 'google search', 'web search', 'check the internet for', 'look on the internet for', 'find online', 'search for']):
                    if 'web_browser' in self.modules:
                        # Extract search query
                        import re
                        patterns = [
                            r'search (?:the web |online |)for (.+)',
                            r'web search for (.+)',
                            r'google search for (.+)',
                            r'look up (.+?)(?: online| on the web)?$',
                            r'check the internet for (.+)',
                            r'look on the internet for (.+)',
                            r'find online (.+)'
                        ]

                        query = None
                        for pattern in patterns:
                            match = re.search(pattern, command_lower)
                            if match:
                                query = match.group(1).strip()
                                break

                        if query:
                            self.logger.info(f"Searching the web for: {query}")
                            search_results = self.modules['web_browser'].search(query)

                            if search_results['success']:
                                response = f"🔍 **Web Search Results for '{query}':**\n\n"
                                for i, res in enumerate(search_results['results'][:5], 1):
                                    response += f"{i}. **{res['title']}**\n"
                                    response += f"   URL: {res['url']}\n"
                                    if res.get('content'):
                                        # We have actual content!
                                        response += f"   **Content:** {res['content'][:300]}...\n"
                                    else:
                                        response += f"   {res.get('snippet', '')[:200]}...\n"
                                    response += "\n"

                                # Store in memory
                                if 'memory' in self.modules:
                                    self.modules['memory'].remember_conversation(command, response)
                                    self.modules['memory'].learn_fact(f"Searched for {query}", 'web_searches')

                                return response
                            else:
                                return f"❌ Search failed: {search_results.get('error', 'Unknown error')}"
                    else:
                        return "Web browser module not initialized. Cannot search the web."

                # GUI creation request
                if any(phrase in command_lower for phrase in ['open window', 'create gui', 'show gui', 'graphical interface', 'visual interface']):
                    if 'gui_manager' in self.modules and 'tennis_scheduler' in self.modules:
                        if 'schedule' in command_lower or 'calendar' in command_lower:
                            self.logger.info("Creating schedule GUI")
                            # Get current schedule
                            schedule_data = self.modules['tennis_scheduler'].get_week_schedule()

                            # Create GUI
                            gui_result = self.modules['gui_manager'].create_schedule_gui(schedule_data)

                            if gui_result['success']:
                                result = f"✅ {gui_result['message']}\n\n"
                                result += "The schedule GUI is now open in a new window. You can:\n"
                                result += "• View your weekly schedule\n"
                                result += "• See today's lessons\n"
                                result += "• Check pending court bookings\n"
                                result += "• Add new lessons"

                                if 'memory' in self.modules:
                                    self.modules['memory'].remember_conversation(command, result)
                                return result
                            else:
                                return f"❌ Failed to create GUI: {gui_result['message']}"
                    else:
                        return "I cannot create GUI windows yet. I need the GUI module to be properly initialized."

                # Scheduling requests
                if any(phrase in command_lower for phrase in ['schedule', 'add lesson', 'book lesson', 'tennis lesson']):
                    if 'tennis_scheduler' in self.modules:
                        # Natural language scheduling
                        if 'add' in command_lower or 'schedule' in command_lower or 'book' in command_lower:
                            self.logger.info("Processing scheduling request")
                            result = self.modules['tennis_scheduler'].add_lesson_from_natural_language(command)

                            if result['success']:
                                response = f"✅ {result['message']}\n\n"
                                if result['needs_court_booking']:
                                    response += "⚠️ **Remember:** You need to call Tennis 13 to book a court for this lesson!\n\n"

                                # Show updated schedule
                                response += self.modules['tennis_scheduler'].get_visual_schedule()

                                if 'memory' in self.modules:
                                    self.modules['memory'].remember_conversation(command, response)
                                return response

                        # Show schedule
                        if 'show' in command_lower or 'display' in command_lower or 'what' in command_lower:
                            schedule = self.modules['tennis_scheduler'].get_visual_schedule()
                            if 'memory' in self.modules:
                                self.modules['memory'].remember_conversation(command, schedule)
                            return schedule

                # Code execution request
                if any(phrase in command_lower for phrase in ['run python', 'execute python', 'run code', 'execute code', 'run script']):
                    if 'code_executor' in self.modules:
                        # Extract code block if present
                        import re
                        code_match = re.search(r'```(?:python)?\n?(.*?)```', command, re.DOTALL)
                        if code_match:
                            code = code_match.group(1)
                            self.logger.info("Executing Python code")
                            success, stdout, stderr = self.modules['code_executor'].execute_python_code(code)

                            if success:
                                result = f"✅ **Code executed successfully:**\n\n```\n{stdout}\n```"
                            else:
                                result = f"❌ **Code execution failed:**\n\n```\n{stderr}\n```"

                            if 'memory' in self.modules:
                                self.modules['memory'].remember_conversation(command, result)
                            return result
                        else:
                            return "Please provide Python code in triple backticks (```) to execute."
                    else:
                        return "I don't have code execution capability initialized."


                # Personal knowledge queries
                if any(phrase in command_lower for phrase in ['what do you know about me', 'my interests', 'my patterns', 'my habits', 'personal insights']):
                    if 'knowledge_extractor' in self.modules:
                        summary = self.modules['knowledge_extractor'].get_personal_summary()
                        response = "**Personal Knowledge Summary:**\n\n"
                        response += f"• Interests identified: {summary['interests_identified']}\n"
                        response += f"• Routines found: {summary['routines_found']}\n"
                        response += f"• Relationships mapped: {summary['relationships_mapped']}\n"
                        response += f"• Locations identified: {summary['locations_identified']}\n"
                        response += f"• Privacy status: {summary['privacy_status']}\n\n"

                        # Get suggestions
                        suggestions = self.modules['knowledge_extractor'].suggest_based_on_patterns()
                        if suggestions:
                            response += "**Personalized Suggestions:**\n"
                            for suggestion in suggestions[:3]:
                                response += f"• {suggestion['suggestion']}\n"

                        if 'memory' in self.modules:
                            self.modules['memory'].remember_conversation(command, response)
                        return response
                    else:
                        return "Personal knowledge extraction is not available."

                # Check for capability gaps and offer to create modules
                gap_analysis = self.modules['capability_manager'].identify_capability_gap(command)
                if not gap_analysis['can_fulfill'] and gap_analysis['can_create_solution']:
                    self.logger.info(f"Capability gap detected: {gap_analysis['missing_capabilities']}")

                    if 'module_creator' in self.modules:
                        response = f"I notice I'm missing some capabilities for your request:\n"
                        response += f"Missing: {', '.join(gap_analysis['missing_capabilities'])}\n\n"
                        response += "Would you like me to create the necessary modules? I can build:\n"
                        for module in gap_analysis['suggested_modules']:
                            response += f"• {module}\n"

                        if 'memory' in self.modules:
                            self.modules['memory'].remember_conversation(command, response)
                        return response

            # Check if this is a system command we can execute directly
            executed = False
            result = None
            error_occurred = False

            if self.command_parser and 'system' in self.modules:
                action_type, params = self.command_parser.parse(command)

                if action_type:
                    # Validate the extracted command before running
                    if action_type == 'run_command' and params.get('command'):
                        # Check if the extracted command makes sense
                        extracted_cmd = params['command']
                        if len(extracted_cmd) < 3 or extracted_cmd.endswith('.'):
                            self.logger.warning(f"Suspicious command extracted: '{extracted_cmd}'")
                            error_occurred = True
                            result = f"❌ Command parsing error: Extracted '{extracted_cmd}' which doesn't look like a valid command.\n\n"

                            # Try to understand what was really wanted
                            if 'self_awareness' in self.modules:
                                result += self.modules['self_awareness'].explain_error(command, f"Invalid command: {extracted_cmd}")
                        else:
                            # Execute the system action
                            result = await self._execute_system_action(action_type, params)
                            executed = True
                    else:
                        # Execute other action types normally
                        result = await self._execute_system_action(action_type, params)
                        executed = True

            # If no pattern matched, check if this LOOKS like a system command
            if not executed and self.intelligent_executor and 'system' in self.modules:
                # Check if this is likely a system request vs conversation
                system_keywords = ['check', 'show', 'list', 'display', 'run', 'execute',
                                 'monitor', 'status', 'info', 'what is', 'what\'s',
                                 'how much', 'how many', 'test', 'scan', 'find']

                command_lower = command.lower()
                is_system_request = any(keyword in command_lower for keyword in system_keywords)

                # Also check for specific system terms
                system_terms = ['cpu', 'gpu', 'memory', 'disk', 'storage', 'network',
                              'battery', 'temperature', 'port', 'process', 'file',
                              'wifi', 'bluetooth', 'usb', 'audio', 'kernel', 'package',
                              'service', 'internet', 'speed']

                has_system_term = any(term in command_lower for term in system_terms)

                # Only use intelligent executor for likely system commands
                if is_system_request or has_system_term:
                    self.logger.info("Looks like a system command, using Intelligent Executor...")
                    result = self.intelligent_executor.execute_intelligent_command(
                        command,
                        self.modules['system']
                    )

                    # Only mark as executed if we got a real result
                    if result and not result.startswith("❌ Couldn't understand"):
                        executed = True

                        # Learn from this execution
                        if 'learning' in self.modules:
                            self.modules['learning'].learn_from_interaction(command, result, True)

            # If we executed something (successfully or not), return the result
            if (executed or error_occurred) and result:
                # Store in memory
                if 'memory' in self.modules:
                    self.modules['memory'].remember_conversation(command, result)

                # If it was an error, add helpful context
                if error_occurred or (result and '❌' in result):
                    result += "\n\n💡 **What you can try:**\n"
                    result += "- Ask me directly about myself (e.g., 'What model do you run on?')\n"
                    result += "- Use 'familiarize yourself with my system' for system discovery\n"
                    result += "- Just chat normally - I understand natural language!\n"

                return result

            # Not a direct system command, process normally
            # Get conversation context from memory
            context = {}
            if 'memory' in self.modules:
                # Get recent conversation history
                recent_memories = self.modules['memory'].get_recent_memories(5)
                if recent_memories:
                    context['history'] = recent_memories

            # Think about the command with context
            thought = await self.think(
                f"User command: {command}\n"
                f"You are PHOENIX, a personal AI assistant running locally on the user's computer.\n"
                f"You have direct system access through the SystemControl module.\n"
                f"Respond conversationally and explain any actions.",
                context=context
            )

            # Check for hallucination before returning response
            if 'capability_manager' in self.modules:
                # Check if the response claims to do something we can't
                hallucination_phrases = [
                    ("send", "email", "I don't have email capability yet"),
                    ("open", "browser", "I cannot control web browsers directly"),
                    ("voice", "speak", "I don't have voice interaction capability"),
                    ("sms", "text message", "I cannot send SMS messages")
                ]

                thought_lower = thought.lower()
                for phrase1, phrase2, correction in hallucination_phrases:
                    if phrase1 in thought_lower and phrase2 in thought_lower:
                        # Check if it's claiming to do it (not saying it can't)
                        if not any(neg in thought_lower for neg in ['cannot', "can't", "don't have", "unable"]):
                            self.logger.warning(f"Caught hallucination about {phrase1} {phrase2}")
                            thought = f"{correction}, but I could create a module for it if you need. Would you like me to do that?"
                            break

            # Store in memory and learn
            success = 'error' not in thought.lower()

            if 'memory' in self.modules:
                self.modules['memory'].remember_conversation(command, thought)

                # Check if user is teaching us something
                if any(phrase in command.lower() for phrase in ['remember that', 'my name is', 'i prefer', 'i like']):
                    # Extract and store user information
                    if 'my name is' in command.lower():
                        name = command.lower().split('my name is')[-1].strip().split()[0].rstrip(',')
                        self.modules['memory'].update_user_profile('name', name.capitalize())
                        self.logger.info(f"Learned user's name: {name}")

                    if 'remember that' in command.lower():
                        fact = command.lower().split('remember that')[-1].strip()
                        self.modules['memory'].learn_fact(fact, 'user_preferences')
                        self.logger.info(f"Learned fact: {fact}")

            # Advanced learning
            if 'learning' in self.modules:
                self.modules['learning'].learn_from_interaction(command, thought, success)

                # Get suggestions if available
                suggestion = self.modules['learning'].suggest_improvement(command)
                if suggestion:
                    thought += f"\n\n{suggestion}"

                # Show predictions for next command
                predictions = self.modules['learning'].predict_next_command(command)
                if predictions and predictions[0][1] > 0.5:  # Show if confidence > 50%
                    thought += "\n\n💡 **You might want to:**"
                    for pred_cmd, confidence in predictions[:3]:
                        thought += f"\n  • {pred_cmd} ({confidence:.0%} likely)"

            return thought

        except Exception as e:
            self.logger.error(f"Error processing command: {e}")
            return f"I encountered an error: {e}"

    async def _execute_system_action(self, action_type: str, params: Dict[str, Any]) -> str:
        """
        Execute a system action and return the result.

        Args:
            action_type: Type of action to execute
            params: Parameters for the action

        Returns:
            Formatted result string
        """
        system = self.modules['system']

        try:
            if action_type == 'system_info':
                # Check BOTH GPU and CPU information
                self.logger.info("Checking full system information...")

                # Get system info
                sys_info = system.get_system_info()
                result = f"🖥️ **System Information:**\n\n"

                # Try to get GPU info
                success, stdout, stderr = system.run_command("nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader")

                if success and stdout:
                    gpu_info = stdout.strip()
                    result += f"**GPU:** {gpu_info.split(',')[0]}\n"
                    result += f"**VRAM:** {gpu_info.split(',')[1]} (Used: {gpu_info.split(',')[2]}, Free: {gpu_info.split(',')[3]})\n"
                    if len(gpu_info.split(',')) > 4:
                        result += f"**GPU Temp:** {gpu_info.split(',')[4].strip()}°C\n"
                        result += f"**GPU Usage:** {gpu_info.split(',')[5].strip()}%\n\n"

                # Add CPU info
                result += f"**CPU:** {sys_info['cpu']['count']} cores @ {sys_info['cpu']['frequency']:.2f} MHz\n"
                result += f"**CPU Usage:** {sys_info['cpu']['percent']}%\n"
                result += f"**RAM:** {sys_info['memory']['total_gb']} GB (Used: {sys_info['memory']['used_gb']} GB)\n"
                result += f"**Disk:** {sys_info['disk']['used_gb']}/{sys_info['disk']['total_gb']} GB used ({sys_info['disk']['percent']}%)"

                return result

            elif action_type == 'cpu_info':
                # Check ONLY CPU information
                self.logger.info("Checking CPU information...")

                # Get detailed CPU info
                success, stdout, stderr = system.run_command("lscpu | grep -E 'Model name|Socket|Core|Thread|CPU MHz|CPU max'")
                sys_info = system.get_system_info()

                result = f"🖥️ **CPU Information:**\n\n"

                if success and stdout:
                    # Parse lscpu output
                    for line in stdout.strip().split('\n'):
                        if 'Model name' in line:
                            result += f"**Model:** {line.split(':')[1].strip()}\n"
                        elif 'Socket' in line:
                            result += f"**Sockets:** {line.split(':')[1].strip()}\n"
                        elif 'Core(s) per socket' in line:
                            result += f"**Cores per Socket:** {line.split(':')[1].strip()}\n"
                        elif 'Thread(s) per core' in line:
                            result += f"**Threads per Core:** {line.split(':')[1].strip()}\n"

                result += f"\n**Total Cores:** {sys_info['cpu']['count']}\n"
                result += f"**Current Frequency:** {sys_info['cpu']['frequency']:.2f} MHz\n"
                result += f"**Current Usage:** {sys_info['cpu']['percent']}%\n"

                # Add per-core usage if available
                try:
                    import psutil
                    per_cpu = psutil.cpu_percent(interval=1, percpu=True)
                    result += f"\n**Per-Core Usage:**\n"
                    for i, usage in enumerate(per_cpu):
                        result += f"  Core {i}: {usage}%\n"
                        if i >= 7:  # Limit to first 8 cores for display
                            result += f"  ... and {len(per_cpu) - 8} more cores\n"
                            break
                except:
                    pass

                return result

            elif action_type == 'gpu_info':
                # Check ONLY GPU information
                self.logger.info("Checking GPU information...")

                result = f"🎮 **GPU Information:**\n\n"

                # Get detailed GPU info
                success, stdout, stderr = system.run_command("nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,power.draw,power.limit --format=csv,noheader")

                if success and stdout:
                    parts = stdout.strip().split(',')
                    result += f"**Model:** {parts[0]}\n"
                    result += f"**VRAM Total:** {parts[1]}\n"
                    result += f"**VRAM Used:** {parts[2]}\n"
                    result += f"**VRAM Free:** {parts[3]}\n"
                    if len(parts) > 4:
                        result += f"**Temperature:** {parts[4].strip()}°C\n"
                        result += f"**Utilization:** {parts[5].strip()}%\n"
                    if len(parts) > 6:
                        result += f"**Power Draw:** {parts[6].strip()}\n"
                        result += f"**Power Limit:** {parts[7].strip()}\n"

                    # Check for multiple GPUs
                    success2, stdout2, stderr2 = system.run_command("nvidia-smi -L")
                    if success2 and stdout2:
                        gpu_count = len(stdout2.strip().split('\n'))
                        if gpu_count > 1:
                            result += f"\n**Note:** You have {gpu_count} GPUs installed"
                else:
                    # Fallback if nvidia-smi doesn't work
                    success, stdout, stderr = system.run_command("lspci | grep -i vga")
                    if success and stdout:
                        result += f"**Detected GPUs:**\n{stdout.strip()}\n\n"
                        result += "⚠️ NVIDIA drivers may not be installed. Install with:\n"
                        result += "`sudo apt install nvidia-driver-535` (or latest version)"
                    else:
                        result += "❌ No GPU detected"

                return result

            elif action_type == 'general_system':
                sys_info = system.get_system_info()
                result = "**System Status:**\n"
                result += f"CPU: {sys_info['cpu']['percent']}% | "
                result += f"Memory: {sys_info['memory']['percent']}% | "
                result += f"Disk: {sys_info['disk']['percent']}%"
                return result

            elif action_type == 'memory_info':
                sys_info = system.get_system_info()
                return f"**Memory:** {sys_info['memory']['used_gb']}/{sys_info['memory']['total_gb']} GB used ({sys_info['memory']['percent']}%)"

            elif action_type == 'disk_info':
                sys_info = system.get_system_info()
                return f"**Disk:** {sys_info['disk']['used_gb']}/{sys_info['disk']['total_gb']} GB used ({sys_info['disk']['percent']}%)"

            elif action_type == 'list_files':
                directory = params.get('directory', '.')
                files = system.list_files(directory)
                if files and not any('error' in f for f in files):
                    result = f"**Files in {directory}:**\n"
                    for f in files[:20]:  # Limit to 20 files
                        icon = '📁' if f['type'] == 'directory' else '📄'
                        result += f"{icon} {f['name']}\n"
                    if len(files) > 20:
                        result += f"... and {len(files) - 20} more files"
                    return result
                return f"Could not list files in {directory}"

            elif action_type == 'process_list':
                processes = system.list_processes()[:10]  # Top 10
                result = "**Top processes by memory:**\n"
                for p in processes:
                    result += f"• {p['name']} (PID: {p['pid']}) - CPU: {p['cpu']}%, Memory: {p['memory']}%\n"
                return result

            elif action_type == 'run_command':
                cmd = params.get('command', '')
                if cmd:
                    success, stdout, stderr = system.run_command(cmd)
                    if success:
                        return f"**Command executed:**\n```\n{stdout}\n```"
                    else:
                        return f"**Command failed:**\n```\n{stderr}\n```"
                return "No command specified"

            else:
                return f"Action type '{action_type}' not implemented yet"

        except Exception as e:
            self.logger.error(f"Error executing system action: {e}")
            return f"Error executing system action: {e}"

    async def run(self):
        """
        Main run loop for the AI system.
        This keeps the AI running and responsive.
        """
        self.is_running = True
        self.logger.info(f"{self.name} is now running!")

        try:
            # Initialize all modules
            await self.initialize_modules()

            # Start background tasks
            # asyncio.create_task(self._monitor_system())
            # asyncio.create_task(self._process_learning())

            # Main interaction loop
            print(f"\n{self.name} is ready! Type 'help' for commands or 'quit' to exit.\n")

            while self.is_running:
                try:
                    # Get user input
                    user_input = input("You: ").strip()

                    # Check for special commands
                    if user_input.lower() in ['quit', 'exit', 'shutdown']:
                        print("Shutting down...")
                        break
                    elif user_input.lower() == 'help':
                        print(self._get_help_text())
                        continue
                    elif user_input.lower() == 'learning report':
                        if 'learning' in self.modules:
                            print(self.modules['learning'].generate_learning_report())
                        else:
                            print("Learning module not available")
                        continue
                    elif 'familiarize' in user_input.lower() or 'discover system' in user_input.lower():
                        if 'discovery' in self.modules:
                            print("\n🔍 Starting system discovery and learning...\n")
                            report = await self.modules['discovery'].full_system_discovery(verbose=True)
                            print(report)
                            print("\n✅ I've learned about your system! I'll remember all of this.")
                        else:
                            print("System discovery module not available")
                        continue
                    elif user_input.lower() == 'automation status':
                        if 'automation' in self.modules:
                            status = self.modules['automation'].get_automation_status()
                            print(f"Automation: {'Running' if status['is_running'] else 'Stopped'}")
                            print(f"Active monitors: {status['active_monitors']}")
                            print(f"File watchers: {status['file_watchers']}")
                        else:
                            print("Automation module not available")
                        continue
                    elif user_input.lower() == 'model status':
                        if 'multi_model' in self.modules:
                            print(self.modules['multi_model'].get_summary())
                        else:
                            print("Multi-Model Intelligence not available")
                        continue
                    elif user_input.lower() == 'learning status':
                        if 'knowledge' in self.modules:
                            summary = self.modules['knowledge'].knowledge_base
                            print("\n📊 Active Learning Status:")
                            print(f"  • Patterns detected: {len(summary.get('command_sequences', []))}")
                            print(f"  • Habits learned: {len(summary.get('workflow_habits', []))}")
                            print(f"  • System knowledge items: {len(summary.get('system', {}))}")
                            insights = self.modules['knowledge'].get_actionable_insights()
                            if insights:
                                print(f"  • Actionable insights: {len(insights)}")
                                for insight in insights[:3]:
                                    print(f"    - {insight['insight']}")
                        else:
                            print("Active Learning not available")
                        continue
                    elif 'predict' in user_input.lower():
                        if 'predictive_model' in self.modules:
                            context = {'directory': os.getcwd()}
                            predictions = self.modules['predictive_model'].predict_next_actions(context)
                            print("\n🔮 Predictions:")
                            for pred in predictions[:3]:
                                print(f"  • {pred.get('target', 'unknown')} (confidence: {pred.get('confidence', 0):.2f})")
                        else:
                            print("Predictive modeling not available")
                        continue

                    # Process the command
                    response = await self.process_command(user_input)
                    print(f"\n{self.name}: {response}\n")

                except KeyboardInterrupt:
                    print("\nInterrupted by user.")
                    break
                except EOFError:
                    # This happens when running without stdin (like in tests)
                    self.logger.info("No input available (EOF), shutting down...")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    print(f"Error: {e}")
                    # Don't continue on critical errors
                    if "EOF" in str(e):
                        break

        finally:
            await self.shutdown()

    def _get_help_text(self) -> str:
        """
        Get help text for user commands.

        Returns:
            Help text string
        """
        help_text = """
        === PHOENIX AI System Commands ===

        General:
        - help: Show this help message
        - status: Show system status
        - quit/exit: Shutdown the system

        System Control:
        - list files in [directory]: List files in a directory
        - run [command]: Execute a system command (with safety checks)
        - show system info: Display CPU, memory, disk usage

        Memory & Learning:
        - my name is [name]: Tell me your name
        - remember that [fact]: Store a fact in memory
        - what do you remember about [topic]: Recall memories
        - show memory stats: Display memory statistics
        - forget [topic]: Remove knowledge about a topic

        Self-Improvement:
        - analyze yourself: Run self-diagnostic
        - improve [module]: Attempt to improve a specific module
        - show capabilities: List current capabilities

        Multi-Model Intelligence:
        - model status: Show available AI models and their specializations
        - switch model [name]: Switch to a specific AI model

        Active Learning:
        - learning status: Show what I've learned about your system
        - predict: Show predictions for your next actions
        - familiarize: Deep scan to learn about your system

        Just talk naturally for general conversation!
        Your conversations are being remembered and I learn from your patterns.
        """

        # Add memory stats if available
        if 'memory' in self.modules:
            stats = self.modules['memory'].get_stats()
            help_text += f"\n        Memory: {stats['total_conversations']} conversations, {stats['total_facts']} facts learned"

        return help_text

    async def shutdown(self):
        """
        Gracefully shutdown the AI system.
        Save state and clean up resources.
        """
        self.logger.info("Shutting down PHOENIX...")
        self.is_running = False

        # Save memory state
        if 'memory' in self.modules:
            self.logger.info("Saving memory to disk...")
            self.modules['memory'].save()

        # Save pattern recognition data
        if 'pattern_engine' in self.modules:
            self.logger.info("Saving patterns to disk...")
            pattern_file = Path('./data/patterns.json')
            self.modules['pattern_engine'].export_patterns(pattern_file)

        # Save habits if available
        if 'habit_learner' in self.modules:
            self.logger.info("Saving habits to disk...")
            habit_file = Path('./data/habits.json')
            self.modules['habit_learner'].export_habits(habit_file)

        # Clean up modules
        for name, module in self.modules.items():
            self.logger.info(f"Shutting down {name} module...")
            # await module.shutdown()

        self.logger.info("PHOENIX shutdown complete.")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status.

        Returns:
            Status dictionary
        """
        uptime = datetime.now() - self.start_time

        return {
            'name': self.name,
            'version': self.version,
            'uptime': str(uptime),
            'mode': self.config['system']['mode'],
            'modules': list(self.modules.keys()),
            'is_running': self.is_running,
            'current_task': self.current_task
        }


async def main():
    """
    Main entry point for running PHOENIX.
    """
    print("""
    ╔═══════════════════════════════════════╗
    ║          PHOENIX AI SYSTEM            ║
    ║   Personal Hybrid Operating Environment   ║
    ║     Network Intelligence eXtension     ║
    ╚═══════════════════════════════════════╝
    """)

    # Create and run Phoenix
    phoenix = PhoenixCore()
    await phoenix.run()


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())