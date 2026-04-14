import matplotlib.pyplot as plt
import numpy as np

def plot_performance(
    system_rewards, 
    system_queues, 
    junction_rewards, 
    junction_queues,
    junction_timers,
    junction_served,
    filename='optimized_traffic_results.png'
):
    """
    Creates a comprehensive 9-graph dashboard for multi-agent traffic analysis.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Multi-Agent Traffic RL — Detailed Performance Dashboard', fontsize=20, fontweight='bold')
    
    def smooth(data, window=20):
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    colors = ['#1D9E75', '#E63946']
    
    # Row 0: Individual Junction Metrics
    # [0, 0] Individual Junction Rewards
    for i in range(2):
        axes[0, 0].plot(smooth(junction_rewards[i]), color=colors[i], label=f'Junction {i}', alpha=0.8)
    axes[0, 0].set_title('Episode Rewards per Junction', fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # [0, 1] Individual Junction Queues
    for i in range(2):
        axes[0, 1].plot(smooth(junction_queues[i]), color=colors[i], label=f'Junction {i}', alpha=0.8)
    axes[0, 1].set_title('Avg Queue Length per Junction', fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Vehicles')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # [0, 2] Signal Duration (Timers)
    for i in range(2):
        axes[0, 2].plot(smooth(junction_timers[i]), color=colors[i], label=f'Junction {i}', alpha=0.8)
    axes[0, 2].set_title('Mean Signal Duration (Timer)', fontweight='bold')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Seconds')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Row 1: System-wide Metrics
    # [1, 0] System-wide Total Reward
    axes[1, 0].plot(smooth(system_rewards), color='#378ADD', linewidth=2)
    axes[1, 0].set_title('System Aggregate Reward', fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Sum of Rewards')
    axes[1, 0].grid(True, alpha=0.3)

    # [1, 1] System-wide Total Queue
    axes[1, 1].plot(smooth(system_queues), color='#BA7517', linewidth=2)
    axes[1, 1].set_title('Network-wide Total Queue', fontweight='bold')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Total Vehicles Waiting')
    axes[1, 1].grid(True, alpha=0.3)

    # [1, 2] Throughput (Vehicles Served)
    for i in range(2):
        axes[1, 2].plot(smooth(junction_served[i]), color=colors[i], label=f'Junction {i}', alpha=0.8)
    axes[1, 2].set_title('Throughput (Vehicles Served)', fontweight='bold')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # Row 2: Advanced Analysis
    # [2, 0] Queue Imbalance (|J0 - J1|)
    q_diff = [abs(q0 - q1) for q0, q1 in zip(junction_queues[0], junction_queues[1])]
    axes[2, 0].plot(smooth(q_diff), color='#9B59B6', linewidth=2)
    axes[2, 0].set_title('Traffic Load Imbalance (|J0 - J1|)', fontweight='bold')
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Vehicle Difference')
    axes[2, 0].grid(True, alpha=0.3)

    # [2, 1] Reward Variance (Stability)
    j0_var = [np.var(junction_rewards[0][max(0, k-10):k+1]) for k in range(len(junction_rewards[0]))]
    axes[2, 1].plot(smooth(j0_var), color='#E67E22', linewidth=2)
    axes[2, 1].set_title('Training Stability (Reward Var)', fontweight='bold')
    axes[2, 1].set_xlabel('Episode')
    axes[2, 1].set_ylabel('Variance')
    axes[2, 1].grid(True, alpha=0.3)

    # [2, 2] Learning Progress (Rolling Averages)
    window = 50
    for i in range(2):
        ma = [np.mean(junction_queues[i][max(0, k-window):k+1]) for k in range(len(junction_queues[i]))]
        axes[2, 2].plot(ma, color=colors[i], label=f'Junction {i}', linewidth=2)
    axes[2, 2].set_title(f'Long-term Progress ({window}-ep MA)', fontweight='bold')
    axes[2, 2].set_xlabel('Episode')
    axes[2, 2].set_ylabel('Avg Queue')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close()
    print(f"9-Graph Dashboard saved to {filename}")

def print_metrics(episode, reward, queue, agents):
    print(f"Ep {episode:4d} | Total Reward: {reward:8.1f} | Avg Queue: {queue:5.1f} | eps: {agents[0].epsilon:.3f}")
