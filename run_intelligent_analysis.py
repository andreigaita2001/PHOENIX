#!/usr/bin/env python3
"""
Run Intelligent Analysis on Personal Data
This actually uses your local LLM to understand your content.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.personal_data_vault import PersonalDataVault
from modules.intelligent_analyzer import IntelligentAnalyzer, OperationHistory, GPUMonitor


async def main():
    """Run intelligent analysis."""

    print("""
╔═══════════════════════════════════════════════════════════════╗
║       PHOENIX - Intelligent Content Analysis                  ║
║                                                               ║
║  This will actually USE your local AI to understand           ║
║  your personal data and extract real insights.                ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    # Initialize
    print("🔧 Initializing systems...")
    vault = PersonalDataVault()
    analyzer = IntelligentAnalyzer(vault)

    # Check GPU
    gpu_info = analyzer.gpu_monitor.get_gpu_info()
    if gpu_info.get('available'):
        print(f"✅ GPU detected: {gpu_info['memory_used_mb']}/{gpu_info['memory_total_mb']} MB used")
        print(f"   Utilization: {gpu_info['utilization']}%")
        print(f"   Temperature: {gpu_info['temperature']}°C")
    else:
        print("⚠️  No GPU detected - will use CPU (slower)")

    print(f"🤖 Using model: {analyzer.model}")

    # Check what we have
    vault.cursor.execute('SELECT COUNT(*) FROM personal_data WHERE category = "emails"')
    email_count = vault.cursor.fetchone()[0]

    print(f"\n📧 Found {email_count:,} emails in vault")

    # Ask how many to analyze
    print("\n" + "="*70)
    print("How many emails should I analyze intelligently?")
    print("="*70)
    print("\nOptions:")
    print("  1) Quick test (10 emails) - ~30 seconds")
    print("  2) Small batch (100 emails) - ~5 minutes")
    print("  3) Medium batch (500 emails) - ~20 minutes")
    print("  4) Large batch (1000 emails) - ~40 minutes")
    print("  5) All emails - VERY LONG")
    print()

    choice = input("Choice [1-5]: ").strip()

    limits = {
        '1': 10,
        '2': 100,
        '3': 500,
        '4': 1000,
        '5': email_count
    }

    limit = limits.get(choice, 100)

    print(f"\n🚀 Starting intelligent analysis of {limit:,} emails...")
    print("    (This will actually use your local AI to understand content)")
    print()

    # Confirm
    response = input("Proceed? (yes/no): ").strip().lower()
    if response != 'yes':
        print("\n❌ Analysis cancelled")
        return

    print("\n" + "="*70)
    print("🧠 INTELLIGENT ANALYSIS IN PROGRESS")
    print("="*70)
    print("\nThis will:")
    print("  • Read email subjects and content")
    print("  • Use your local Qwen 2.5 14B to understand meaning")
    print("  • Extract topics, people, interests")
    print("  • Build semantic understanding")
    print("  • Track GPU usage and performance")
    print()

    # Run analysis
    results = await analyzer.analyze_emails_intelligent(
        limit=limit,
        batch_size=5  # Process 5 at a time for memory efficiency
    )

    # Show results
    print("\n" + "="*70)
    print("✨ ANALYSIS COMPLETE!")
    print("="*70)

    print(f"\n📊 Results:")
    print(f"   • Emails analyzed: {results['emails_analyzed']:,}")
    print(f"   • Insights extracted: {results['insights_extracted']:,}")
    print(f"   • Processing time: {results['processing_time_ms']/1000:.1f} seconds")

    print(f"\n🔍 Top Topics Found:")
    for topic, count in results['topics_found'][:10]:
        print(f"   • {topic}: {count} occurrences")

    if results['people_mentioned']:
        print(f"\n👥 People/Organizations Mentioned:")
        for person, count in results['people_mentioned'][:10]:
            print(f"   • {person}: {count} times")

    if results['interests_identified']:
        print(f"\n💡 Interests Identified:")
        for interest, count in results['interests_identified']:
            print(f"   • {interest}: {count} mentions")

    # Get all insights
    print("\n" + "="*70)
    print("📝 Sample Insights:")
    print("="*70)

    insights = analyzer.get_all_insights()[:10]
    for i, insight in enumerate(insights, 1):
        print(f"\n{i}. [{insight['category']}] {insight['content']}")
        if insight['metadata'].get('topics'):
            print(f"   Topics: {', '.join(insight['metadata']['topics'][:3])}")

    # Operation history stats
    print("\n" + "="*70)
    print("📈 Operation Statistics:")
    print("="*70)

    stats = analyzer.history.get_statistics()
    print(f"\n   • Total operations: {stats['total_operations']}")
    print(f"   • Success rate: {stats['success_rate']*100:.1f}%")
    print(f"   • GPU operations: {stats['gpu_operations']} ({stats['gpu_usage_rate']*100:.1f}%)")

    if stats['model_performance']:
        print(f"\n   Model Performance:")
        for model, perf in stats['model_performance'].items():
            print(f"      • {model}:")
            print(f"         - {perf['avg_tokens_per_sec']:.1f} tokens/sec")
            print(f"         - {perf['avg_gpu_util']:.1f}% GPU utilization")
            print(f"         - Used {perf['total_uses']} times")

    # Save report
    report_path = Path.home() / '.phoenix_vault' / 'analysis_report.json'
    import json
    with open(report_path, 'w') as f:
        json.dump({
            'results': results,
            'insights_sample': insights[:20],
            'statistics': stats
        }, f, indent=2)

    print(f"\n💾 Full report saved to: {report_path}")

    print("\n" + "="*70)
    print("✅ INTELLIGENT ANALYSIS COMPLETE")
    print("="*70)
    print("\nNow PHOENIX actually understands your content!")
    print("You can query your insights in PHOENIX with:")
    print("  • 'show my interests'")
    print("  • 'what topics am I interested in?'")
    print("  • 'analyze my communication patterns'")
    print()


if __name__ == "__main__":
    asyncio.run(main())
