"""
Visualize Federated Learning Training Results
Creates plots showing model performance over rounds
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_history(filepath='checkpoints/training_history.json'):
    """Load training history from JSON file"""
    with open(filepath, 'r') as f:
        history = json.load(f)
    return history

def plot_training_curves(history, save_path='checkpoints/training_curves.png'):
    """
    Create comprehensive training visualization
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Federated Learning Training Results', fontsize=16, fontweight='bold')
    
    rounds = history['rounds']
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    ax1.plot(rounds, history['train_loss'], 'b-o', linewidth=2, markersize=8, label='Training Loss')
    ax1.set_xlabel('FL Round', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Over Rounds', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add value labels on points
    for i, (r, loss) in enumerate(zip(rounds, history['train_loss'])):
        ax1.annotate(f'{loss:.3f}', 
                    xy=(r, loss), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # Plot 2: Training Accuracy
    ax2 = axes[0, 1]
    train_acc_pct = [acc * 100 for acc in history['train_accuracy']]
    ax2.plot(rounds, train_acc_pct, 'g-o', linewidth=2, markersize=8, label='Training Accuracy')
    ax2.set_xlabel('FL Round', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training Accuracy Over Rounds', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 100])
    
    # Add value labels
    for i, (r, acc) in enumerate(zip(rounds, train_acc_pct)):
        ax2.annotate(f'{acc:.1f}%', 
                    xy=(r, acc), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
    
    # Plot 3: Evaluation Metrics
    ax3 = axes[1, 0]
    if history['eval_loss']:
        eval_acc_pct = [acc * 100 for acc in history['eval_accuracy']]
        ax3.plot(rounds, history['eval_loss'], 'r-o', linewidth=2, markersize=8, label='Eval Loss')
        ax3.set_xlabel('FL Round', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12, color='r')
        ax3.tick_params(axis='y', labelcolor='r')
        ax3.set_title('Evaluation Loss Over Rounds', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add secondary y-axis for accuracy
        ax3_twin = ax3.twinx()
        ax3_twin.plot(rounds, eval_acc_pct, 'b-s', linewidth=2, markersize=8, label='Eval Accuracy')
        ax3_twin.set_ylabel('Accuracy (%)', fontsize=12, color='b')
        ax3_twin.tick_params(axis='y', labelcolor='b')
        ax3_twin.set_ylim([0, 100])
        
        # Combine legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
    
    # Plot 4: F1 Score
    ax4 = axes[1, 1]
    if history['eval_f1_score']:
        ax4.plot(rounds, history['eval_f1_score'], 'm-o', linewidth=2, markersize=8, label='F1 Score')
        ax4.set_xlabel('FL Round', fontsize=12)
        ax4.set_ylabel('F1 Score', fontsize=12)
        ax4.set_title('F1 Score Over Rounds', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        ax4.set_ylim([0, 1])
        
        # Add value labels
        for i, (r, f1) in enumerate(zip(rounds, history['eval_f1_score'])):
            ax4.annotate(f'{f1:.3f}', 
                        xy=(r, f1), 
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='plum', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training curves saved to: {save_path}")
    plt.show()

def print_summary(history):
    """Print comprehensive training summary"""
    
    print("\n" + "="*70)
    print("📊 FEDERATED LEARNING RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n🔄 Training Configuration:")
    print(f"   Total FL Rounds: {len(history['rounds'])}")
    print(f"   Hospitals Participated: {history['num_clients'][-1]}")
    
    print(f"\n📈 Training Performance:")
    print(f"   Initial Train Loss: {history['train_loss'][0]:.4f}")
    print(f"   Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"   Loss Reduction: {(history['train_loss'][0] - history['train_loss'][-1]):.4f} ({((history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0] * 100):.1f}%)")
    
    print(f"\n   Initial Train Accuracy: {history['train_accuracy'][0]*100:.2f}%")
    print(f"   Final Train Accuracy: {history['train_accuracy'][-1]*100:.2f}%")
    print(f"   Accuracy Improvement: {(history['train_accuracy'][-1] - history['train_accuracy'][0])*100:.2f}%")
    
    if history['eval_accuracy']:
        print(f"\n📊 Evaluation Performance:")
        print(f"   Initial Eval Accuracy: {history['eval_accuracy'][0]*100:.2f}%")
        print(f"   Final Eval Accuracy: {history['eval_accuracy'][-1]*100:.2f}%")
        print(f"   Accuracy Improvement: {(history['eval_accuracy'][-1] - history['eval_accuracy'][0])*100:.2f}%")
        
        print(f"\n   Initial F1 Score: {history['eval_f1_score'][0]:.4f}")
        print(f"   Final F1 Score: {history['eval_f1_score'][-1]:.4f}")
        print(f"   F1 Improvement: {(history['eval_f1_score'][-1] - history['eval_f1_score'][0]):.4f}")
    
    print(f"\n🎯 Best Performance:")
    best_train_acc_idx = np.argmax(history['train_accuracy'])
    print(f"   Best Train Accuracy: {history['train_accuracy'][best_train_acc_idx]*100:.2f}% (Round {history['rounds'][best_train_acc_idx]})")
    
    if history['eval_accuracy']:
        best_eval_acc_idx = np.argmax(history['eval_accuracy'])
        print(f"   Best Eval Accuracy: {history['eval_accuracy'][best_eval_acc_idx]*100:.2f}% (Round {history['rounds'][best_eval_acc_idx]})")
        
        best_f1_idx = np.argmax(history['eval_f1_score'])
        print(f"   Best F1 Score: {history['eval_f1_score'][best_f1_idx]:.4f} (Round {history['rounds'][best_f1_idx]})")
    
    print("\n" + "="*70)
    
    # Per-round breakdown
    print("\n📋 Round-by-Round Performance:")
    print("-"*70)
    print(f"{'Round':<8} {'Train Loss':<12} {'Train Acc':<12} {'Eval Acc':<12} {'F1 Score':<10}")
    print("-"*70)
    
    for i, r in enumerate(history['rounds']):
        eval_acc = f"{history['eval_accuracy'][i]*100:.2f}%" if i < len(history['eval_accuracy']) else "N/A"
        f1 = f"{history['eval_f1_score'][i]:.4f}" if i < len(history['eval_f1_score']) else "N/A"
        print(f"{r:<8} {history['train_loss'][i]:<12.4f} {history['train_accuracy'][i]*100:<11.2f}% {eval_acc:<12} {f1:<10}")
    
    print("-"*70 + "\n")

def main():
    """Main visualization function"""
    
    print("\n" + "="*70)
    print("📊 FEDERATED LEARNING RESULTS VISUALIZATION")
    print("="*70 + "\n")
    
    # Check if history file exists
    history_path = Path('checkpoints/training_history.json')
    if not history_path.exists():
        print("❌ Error: training_history.json not found!")
        print("   Make sure you've run the FL simulation first.")
        return
    
    # Load history
    print("📂 Loading training history...")
    history = load_training_history()
    
    # Print summary
    print_summary(history)
    
    # Create visualizations
    print("\n📊 Generating plots...")
    try:
        plot_training_curves(history)
        print("✅ Visualization complete!")
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        print("   Make sure matplotlib is installed: pip install matplotlib")
    
    print("\n💡 Tips:")
    print("   - Training curves show model improvement over FL rounds")
    print("   - Higher accuracy = better predictions")
    print("   - Lower loss = better model fit")
    print("   - F1 score balances precision and recall (important for imbalanced data)")
    print("   - Your model learned from 3 hospitals without sharing raw patient data!")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()