#!/usr/bin/env python3
"""
Test script for PHOENIX Active Learning System.
Tests system scanning, pattern recognition, habit learning, and knowledge consolidation.
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add PHOENIX to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.active_learning.system_scanner import SystemScanner
from modules.active_learning.pattern_recognition import PatternRecognitionEngine
from modules.active_learning.habit_learning import HabitLearner
from modules.active_learning.predictive_modeling import PredictiveModel
from modules.active_learning.knowledge_consolidation import KnowledgeConsolidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ActiveLearningTest")


async def test_active_learning_system():
    """Test the complete active learning system."""

    logger.info("="*60)
    logger.info("PHOENIX Active Learning System Test")
    logger.info("="*60)

    # Initialize all components
    logger.info("\n🔧 Initializing Active Learning Components...")

    scanner = SystemScanner()
    pattern_engine = PatternRecognitionEngine()
    habit_learner = HabitLearner(pattern_engine)
    predictive_model = PredictiveModel(pattern_engine, habit_learner)
    consolidator = KnowledgeConsolidator(
        scanner, pattern_engine, habit_learner, predictive_model
    )

    # Test 1: System Scanning
    logger.info("\n📡 Testing System Scanner...")
    logger.info("  Performing quick system scan...")

    scan_result = await scanner.full_system_scan()

    logger.info(f"  ✓ OS: {scan_result.get('os_info', {}).get('system', 'Unknown')}")
    logger.info(f"  ✓ CPU: {scan_result.get('hardware', {}).get('cpu_model', 'Unknown')}")
    logger.info(f"  ✓ Memory: {scan_result.get('hardware', {}).get('total_memory_gb', 0):.1f} GB")

    # Scan for projects (limited to /home/bone for speed)
    logger.info("\n  Scanning for projects...")
    projects = scan_result.get('projects', [])

    if projects:
        logger.info(f"  ✓ Found {len(projects)} projects:")
        for project in projects[:3]:  # Show first 3
            logger.info(f"    • {project.get('name', 'Unknown')}: {project.get('path', '')}")
    else:
        logger.info("  • No projects found in quick scan")

    # Test 2: Pattern Recognition
    logger.info("\n🔍 Testing Pattern Recognition...")

    # Simulate some events
    test_events = [
        {'type': 'command', 'data': {'command': 'git status', 'directory': '/home/bone/PHOENIX'}},
        {'type': 'command', 'data': {'command': 'git add .', 'directory': '/home/bone/PHOENIX'}},
        {'type': 'command', 'data': {'command': 'git commit -m "test"', 'directory': '/home/bone/PHOENIX'}},
        {'type': 'file_access', 'data': {'path': '/home/bone/PHOENIX/README.md', 'action': 'read'}},
        {'type': 'file_access', 'data': {'path': '/home/bone/PHOENIX/README.md', 'action': 'edit'}},
        {'type': 'tool_use', 'data': {'tool': 'vscode', 'context': 'editing python files'}},
        {'type': 'error', 'data': {'type': 'FileNotFoundError', 'context': 'missing config'}},
    ]

    for event in test_events:
        pattern_engine.record_event(event['type'], event['data'])

    # Repeat the command sequence to create a pattern
    for _ in range(3):
        pattern_engine.record_event('command', {'command': 'git status'})
        pattern_engine.record_event('command', {'command': 'git add .'})
        pattern_engine.record_event('command', {'command': 'git commit -m "test"'})

    logger.info("  ✓ Recorded test events")

    # Get predictions
    current_context = {
        'directory': '/home/bone/PHOENIX',
        'recent_commands': ['git status', 'git add .']
    }

    predictions = pattern_engine.predict_next_action(current_context)

    if predictions:
        logger.info(f"  ✓ Generated {len(predictions)} predictions:")
        for pred in predictions[:2]:
            logger.info(f"    • {pred['action']} (confidence: {pred['confidence']:.2f})")
    else:
        logger.info("  • No predictions yet (need more data)")

    # Get insights
    insights = pattern_engine.get_insights()

    if insights['most_used_commands']:
        logger.info(f"  ✓ Most used commands: {insights['most_used_commands'][:3]}")

    # Test 3: Habit Learning
    logger.info("\n🧠 Testing Habit Learning...")

    # Learn from patterns
    habit_learner.learn_from_patterns(pattern_engine.patterns)

    # Generate automations
    automations = habit_learner.generate_automations()

    if automations:
        logger.info(f"  ✓ Generated {len(automations)} automation suggestions:")
        for auto in automations[:2]:
            logger.info(f"    • {auto['name']}: {auto['type']}")
    else:
        logger.info("  • No automations yet (patterns need more occurrences)")

    # Test habit suggestions
    suggestions = habit_learner.suggest_next_action(current_context)

    if suggestions:
        logger.info(f"  ✓ Habit-based suggestions: {len(suggestions)}")
        for sugg in suggestions[:2]:
            logger.info(f"    • {sugg['action']} ({sugg['type']})")

    # Get habit summary
    habit_summary = habit_learner.get_habit_summary()
    logger.info(f"  ✓ Learned habits summary:")
    logger.info(f"    • Workflows: {habit_summary['total_workflows']}")
    logger.info(f"    • Daily routines: {habit_summary['daily_routines']}")
    logger.info(f"    • Tool preferences: {habit_summary['tool_preferences']}")

    # Test 4: Predictive Modeling
    logger.info("\n🔮 Testing Predictive Modeling...")

    # Test predictions
    predictions = predictive_model.predict_next_actions(
        {
            'directory': '/home/bone/PHOENIX',
            'recent_actions': ['edit_file', 'run_test', 'commit_code'],
            'last_action_time': datetime.now() - timedelta(minutes=5)
        },
        horizon=3
    )

    if predictions:
        logger.info(f"  ✓ Next action predictions ({len(predictions)}):")
        for pred in predictions[:3]:
            logger.info(f"    • {pred.get('target', 'unknown')} ({pred.get('type', '')}: {pred.get('confidence', 0):.2f})")

    # Test resource predictions
    resource_needs = predictive_model.predict_resource_needs(3600)
    logger.info(f"  ✓ Resource predictions for next hour:")
    logger.info(f"    • CPU: {resource_needs['cpu']['average']}% avg, {resource_needs['cpu']['peak']}% peak")
    logger.info(f"    • Memory: {resource_needs['memory']['average']} MB avg")

    # Get prediction summary
    pred_summary = predictive_model.get_prediction_summary()
    logger.info(f"  ✓ Prediction accuracy: {pred_summary['overall_accuracy']:.1%}")

    # Test 5: Knowledge Consolidation
    logger.info("\n💡 Testing Knowledge Consolidation...")

    # Consolidate all knowledge
    consolidation_summary = consolidator.consolidate_all_knowledge()

    logger.info(f"  ✓ Consolidation complete:")
    logger.info(f"    • Total items: {consolidation_summary['total_items']}")
    logger.info(f"    • Relationships: {consolidation_summary['relationships']}")
    logger.info(f"    • Insights: {consolidation_summary['insights']}")

    # Get actionable insights
    actionable = consolidator.get_actionable_insights()

    if actionable:
        logger.info(f"  ✓ Actionable insights ({len(actionable)}):")
        for insight in actionable[:2]:
            logger.info(f"    • {insight['category']}: {insight['insight']}")

    # Quality metrics
    quality = consolidation_summary['quality_metrics']
    logger.info(f"  ✓ Knowledge quality:")
    logger.info(f"    • Accuracy: {quality['accuracy']:.1%}")
    logger.info(f"    • Completeness: {quality['completeness']:.1%}")
    logger.info(f"    • Consistency: {quality['consistency']:.1%}")
    logger.info(f"    • Relevance: {quality['relevance']:.1%}")

    # Test 6: Integration Test
    logger.info("\n🔗 Testing Component Integration...")

    # Simulate a learning cycle
    logger.info("  Simulating learning cycle...")

    # Record more realistic events
    realistic_events = [
        # Morning routine
        {'hour': 9, 'commands': ['check_email', 'review_tasks', 'open_ide']},
        {'hour': 10, 'commands': ['git pull', 'npm install', 'npm test']},
        {'hour': 11, 'commands': ['edit_code', 'run_tests', 'git commit']},
        # Afternoon routine
        {'hour': 14, 'commands': ['review_prs', 'code_review', 'merge_pr']},
        {'hour': 15, 'commands': ['write_docs', 'update_readme', 'git push']},
    ]

    for event_set in realistic_events:
        # Simulate time-based events
        for cmd in event_set['commands']:
            pattern_engine.record_event('command', {'command': cmd})

        # Create time pattern
        pattern_engine.patterns['time_patterns'][event_set['hour']] = {
            'common_activity': event_set['commands'][0],
            'frequency': 5,
            'confidence': 0.8
        }

    # Learn from new patterns
    habit_learner.learn_from_patterns(pattern_engine.patterns)

    # Make predictions based on learned behavior
    context_morning = {'directory': '/home/bone/PHOENIX', 'hour': 10}
    morning_predictions = predictive_model.predict_next_actions(context_morning)

    if morning_predictions:
        logger.info(f"  ✓ Morning predictions based on learning:")
        for pred in morning_predictions[:2]:
            logger.info(f"    • {pred.get('target', '')} at 10:00")

    # Final consolidation
    final_summary = consolidator.consolidate_all_knowledge()
    logger.info(f"  ✓ Final knowledge base: {final_summary['total_items']} items")

    # Export test
    logger.info("\n💾 Testing Export/Import...")

    # Export all learned data
    export_dir = Path("/tmp/phoenix_active_learning_export")
    export_dir.mkdir(exist_ok=True)

    # Export patterns
    pattern_engine.export_patterns(export_dir / "patterns.json")
    logger.info(f"  ✓ Exported patterns to {export_dir / 'patterns.json'}")

    # Export habits
    habit_learner.export_habits(export_dir / "habits.json")
    logger.info(f"  ✓ Exported habits to {export_dir / 'habits.json'}")

    # Export knowledge
    consolidator.export_knowledge(export_dir / "knowledge.json")
    logger.info(f"  ✓ Exported knowledge to {export_dir / 'knowledge.json'}")

    # System capabilities summary
    logger.info("\n" + "="*60)
    logger.info("🎯 Active Learning System Capabilities:")
    logger.info("  ✓ Autonomous system exploration and discovery")
    logger.info("  ✓ Pattern recognition in user behavior")
    logger.info("  ✓ Habit learning and automation generation")
    logger.info("  ✓ Predictive modeling for anticipating needs")
    logger.info("  ✓ Knowledge consolidation and insight generation")
    logger.info("  ✓ Continuous learning and adaptation")

    logger.info("\n📊 System is learning about:")
    logger.info("  • Your project structure and tools")
    logger.info("  • Your workflow patterns and habits")
    logger.info("  • Your preferred working hours")
    logger.info("  • Common command sequences")
    logger.info("  • File access patterns")
    logger.info("  • Error patterns to avoid")

    logger.info("\n✅ Active Learning System Test Complete!")

    return True


async def main():
    """Main entry point."""
    try:
        success = await test_active_learning_system()

        if success:
            logger.info("\n🎉 PHOENIX Active Learning is ready!")
            logger.info("The system will now continuously learn and adapt to your behavior")
            logger.info("\nNext steps:")
            logger.info("1. Let PHOENIX observe your daily workflows")
            logger.info("2. Review generated automation suggestions")
            logger.info("3. Provide feedback on predictions")
            logger.info("4. Watch as PHOENIX becomes more helpful over time")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(main())