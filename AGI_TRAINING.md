# PHOENIX AGI Training System

## What This Is

Your local AGI (Qwen 2.5 32B) that **actually executes terminal commands autonomously**, learning from both successes and Claude's demonstrations.

## How It Works

```
You input task → AGI generates command → PHOENIX executes it → Result validated
                      ↓ Success                    ↓ Failure
              Pattern stored in KB         Claude demonstrates correct way
```

## Proven Results

**First Training Session:**
- ✅ 5/5 tasks completed autonomously (100% autonomy rate)
- ✅ Real commands executed via PHOENIX SystemControl
- ✅ 5 patterns learned and stored
- ✅ 0 assists needed from Claude

## Components

### 1. Direct Chat Interface (`agi_chat.py`)
**This is what you use daily**

```bash
cd /home/bone/PHOENIX
./agi_chat.py
```

Input your tasks, AGI executes them. Examples:
- "List all Python files in my project"
- "Find files modified in the last hour"
- "Show disk usage of /home/bone"
- "Count lines of code in src/"

### 2. Structured Training (`training/agi_teacher.py`)
Systematic teaching sessions with exercises

```bash
cd /home/bone/PHOENIX
python3 training/agi_teacher.py
```

### 3. Knowledge Base (`training/agi_knowledge.json`)
Stores successful task→command mappings:
- Task descriptions
- Correct commands
- Output samples
- Timestamps

### 4. Progress Tracking (`training/agi_progress.json`)
Tracks autonomy metrics:
- Total tasks attempted
- Autonomous successes
- Claude assists needed
- Overall autonomy rate

## Architecture

**Local AGI (Qwen 2.5 32B)**
- Runs on your RTX GPUs (19GB VRAM)
- Generates commands from task descriptions
- Learns from successful patterns

**PHOENIX SystemControl**
- Proven working command execution
- Safety checks (allowed/forbidden directories)
- Real subprocess execution

**Learning Loop**
1. AGI attempts task autonomously
2. PHOENIX executes and validates
3. Success → Store pattern in knowledge base
4. Failure → Claude demonstrates correct way
5. Repeat → Autonomy rate increases over time

## Goal: Progressive Autonomy

```
Session 1:  100% AGI (basic tasks) ✅
Session 2:  Harder tasks, some Claude assists
Session 3:  More autonomous as patterns learned
...
Final:      100% AGI autonomous, 0% Claude needed
```

## Current Status

- **Autonomy Rate:** 100% (5/5 tasks)
- **Knowledge Base:** 5 patterns learned
- **Total Tasks:** 5 completed
- **Level:** 1 (Basic Commands)

## Comparison: PHOENIX-local-agi vs PHOENIX

**PHOENIX-local-agi (abandoned):**
- ❌ LLM just generated text about commands
- ❌ Tool calling interface broken
- ❌ No actual execution

**PHOENIX (current):**
- ✅ SystemControl actually executes commands
- ✅ Real subprocess calls with output
- ✅ Proven working in production
- ✅ Safety checks built-in

## Usage Examples

**Direct Chat:**
```bash
./agi_chat.py
📝 Task: Find all TODO comments in my code
🤖 Asking local AGI...
   AGI responded: grep -r "TODO" --include="*.py"
✅ AGI succeeded autonomously!
```

**Training Session:**
```bash
python3 training/agi_teacher.py
[Runs 5 exercises, tracks autonomy, stores patterns]
```

**Check Progress:**
```bash
./agi_chat.py
status
📊 AGI STATUS:
   Autonomy Rate: 100.0%
   Autonomous Tasks: 5
   Claude Assists: 0
   Total Tasks: 5
```

## What Makes This Different

**Traditional AI assistants:**
- You ask → AI generates text → You copy/paste command → You execute
- AI never actually does anything

**PHOENIX AGI:**
- You ask → AGI generates command → **AGI executes it** → Result returned
- Real autonomous execution
- Learns from every success
- Progressive independence from Claude

## Safety

- Allowed directories: `/home/bone`
- Forbidden directories: `/etc`, `/sys`, `/proc`
- File size limit: 100MB
- Command timeout: 30s
- All execution via PHOENIX's proven SystemControl

## Next Steps

1. Use `agi_chat.py` for your daily tasks
2. AGI will handle what it knows autonomously
3. Claude demonstrates when AGI encounters new patterns
4. Knowledge base grows
5. Autonomy rate increases
6. Eventually: Fully autonomous terminal mastery

---

**Status:** ✅ Working and proven with 100% autonomy on basic tasks
**Goal:** Full terminal autonomy with 0% Claude dependence
**Progress:** 5/∞ patterns learned, constantly improving
