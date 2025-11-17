# PHOENIX - Personal AI System

## Vision
A self-improving, locally-run AI system that serves as both system overseer and personal assistant.

## Core Capabilities
1. **System Control** - Manage files, processes, and system settings
2. **Self-Improvement** - Learn from interactions and modify its own code
3. **Learning & Memory** - Remember preferences and adapt behavior
4. **Automation** - Handle routine tasks autonomously

## Architecture

### Brain (LLM Core)
- Powered by Ollama with Qwen 2.5 14B model
- Handles reasoning and decision-making
- Generates responses and actions

### Memory Systems
- **Vector Database**: ChromaDB for semantic search
- **Knowledge Graph**: NetworkX for relationships
- **Persistent Storage**: SQLite for structured data

### Safety Mechanisms
- **Sandboxing**: All code modifications tested in isolation
- **Rollback**: Automatic reversion on errors
- **Audit Trail**: Complete logging of all actions
- **Permission System**: Graduated access levels

### Modules
- **System Control**: File operations, process management
- **Self-Improvement**: Code analysis and modification
- **Learning**: Pattern recognition and adaptation
- **Automation**: Scheduled tasks and monitoring

## Safety First
- Every action is logged
- All code changes are versioned
- Rollback capability for all modifications
- Human approval for critical operations

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Configure Ollama: `ollama pull qwen2.5:14b`
3. Initialize: `python phoenix.py --init`
4. Run: `python phoenix.py`

## Project Structure
```
PHOENIX/
├── core/           # Core AI engine
├── modules/        # Feature modules
├── memory/         # Memory systems
├── safety/         # Safety mechanisms
├── config/         # Configuration files
├── tests/          # Test suite
├── data/           # Persistent data
└── logs/           # System logs
```