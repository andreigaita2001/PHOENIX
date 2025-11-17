#!/usr/bin/env python3
"""
View PHOENIX Operation History
See everything PHOENIX has done and optimize performance.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from modules.intelligent_analyzer import OperationHistory, GPUMonitor


def main():
    """View operation history and statistics."""

    print("""
╔═══════════════════════════════════════════════════════════════╗
║         PHOENIX - Operation History Viewer                    ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    # Initialize
    history_path = Path.home() / '.phoenix_vault' / 'operation_history.db'

    if not history_path.exists():
        print("❌ No operation history found yet.")
        print("   Run some operations first (like intelligent analysis)")
        return

    history = OperationHistory(history_path)
    gpu = GPUMonitor()

    # Current GPU status
    print("\n" + "="*70)
    print("🎮 GPU Status:")
    print("="*70)

    gpu_info = gpu.get_gpu_info()
    if gpu_info.get('available'):
        print(f"\n   ✅ GPU Available")
        print(f"   • Memory: {gpu_info['memory_used_mb']:,} MB / {gpu_info['memory_total_mb']:,} MB")
        print(f"   • Free: {gpu_info['memory_total_mb'] - gpu_info['memory_used_mb']:,} MB")
        print(f"   • Utilization: {gpu_info['utilization']}%")
        print(f"   • Temperature: {gpu_info['temperature']}°C")

        if gpu.can_use_large_model():
            print(f"\n   ✅ Can run large models (14B parameters)")
        else:
            print(f"\n   ⚠️  Limited memory - recommend smaller models")
    else:
        print(f"\n   ❌ No GPU detected - using CPU only")

    # Statistics
    print("\n" + "="*70)
    print("📊 Operation Statistics:")
    print("="*70)

    stats = history.get_statistics()

    print(f"\n   Overall:")
    print(f"   • Total operations: {stats['total_operations']:,}")
    print(f"   • Success rate: {stats['success_rate']*100:.1f}%")
    print(f"   • GPU operations: {stats['gpu_operations']:,} ({stats['gpu_usage_rate']*100:.1f}%)")

    if stats['operation_durations']:
        print(f"\n   Average Duration by Type:")
        for op_type, info in stats['operation_durations'].items():
            print(f"   • {op_type}:")
            print(f"      - Average: {info['avg_ms']:.0f}ms")
            print(f"      - Count: {info['count']}")

    if stats['model_performance']:
        print(f"\n   Model Performance:")
        for model, perf in stats['model_performance'].items():
            print(f"   • {model}:")
            print(f"      - {perf['avg_tokens_per_sec']:.1f} tokens/second")
            print(f"      - {perf['avg_gpu_util']:.1f}% GPU utilization")
            print(f"      - Used {perf['total_uses']} times")

    # Recent operations
    print("\n" + "="*70)
    print("📜 Recent Operations (last 20):")
    print("="*70)

    recent = history.get_recent_operations(limit=20)

    if recent:
        for op in recent:
            status = "✅" if op['success'] else "❌"
            gpu_mark = "🎮" if op.get('model') else "💻"

            timestamp = datetime.fromisoformat(op['timestamp']).strftime("%Y-%m-%d %H:%M:%S")

            print(f"\n   {status} {gpu_mark} [{timestamp}]")
            print(f"      Type: {op['type']}")
            print(f"      Module: {op['module']}")
            print(f"      Duration: {op['duration_ms']}ms")
            if op.get('model'):
                print(f"      Model: {op['model']}")
            if op.get('error'):
                print(f"      Error: {op['error']}")
    else:
        print("\n   No operations recorded yet")

    print("\n" + "="*70)
    print("💡 Tips:")
    print("="*70)
    print("\n   • This history helps debug and optimize PHOENIX")
    print("   • GPU operations are faster but use more power")
    print("   • Monitor temperature if running heavy workloads")
    print("   • History stored at:", history_path)
    print()


if __name__ == "__main__":
    main()
