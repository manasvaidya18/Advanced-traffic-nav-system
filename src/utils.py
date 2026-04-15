# import library for plotting graphs
import matplotlib.pyplot as plt   # used to draw graphs

# import library for math operations
import numpy as np                # used for average, variance, etc


# main function to create 9 graphs dashboard
def plot_performance(
    system_rewards,        # total reward of whole traffic system
    system_queues,         # total vehicles waiting in system
    junction_rewards,      # rewards of each junction (2 agents)
    junction_queues,       # queue length at each junction
    junction_timers,       # signal timing (green light duration)
    junction_served,       # number of vehicles passed
    filename='optimized_traffic_results.png'  # output file name
):
    """
    This function creates 9 graphs to analyze traffic RL performance
    """

    # create 3x3 grid (9 graphs)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    # IMPORTANT: creates layout for dashboard

    # set main title for all graphs
    fig.suptitle(
        'Multi-Agent Traffic RL — Detailed Performance Dashboard',
        fontsize=20,
        fontweight='bold'
    )
    # IMPORTANT: title of whole output

    
    # helper function to smooth graph data
    def smooth(data, window=20):
        # IMPORTANT: removes noise from graph

        # if data is small, return original
        if len(data) < window:
            return data

        # moving average formula
        return np.convolve(
            data,                        # input data
            np.ones(window)/window,      # averaging window
            mode='valid'                 # valid output
        )
        # IMPORTANT: smoothing logic


    # colors for two junctions
    colors = ['#1D9E75', '#E63946']
    # just for visualization


    # ================= ROW 0 =================
    # Individual Junction Metrics

    # Graph 1: rewards of each junction
    for i in range(2):   # loop for 2 junctions
        axes[0, 0].plot(
            smooth(junction_rewards[i]),   # smoothed rewards
            color=colors[i],               # color of line
            label=f'Junction {i}',         # label name
            alpha=0.8                     # transparency
        )

    axes[0, 0].set_title('Episode Rewards per Junction', fontweight='bold')
    axes[0, 0].set_xlabel('Episode')   # x-axis = episodes
    axes[0, 0].set_ylabel('Total Reward')  # y-axis = reward
    axes[0, 0].legend()  # show labels
    axes[0, 0].grid(True, alpha=0.3)  # grid for clarity


    # Graph 2: queue length of each junction
    for i in range(2):
        axes[0, 1].plot(
            smooth(junction_queues[i]),  # smoothed queue
            color=colors[i],
            label=f'Junction {i}',
            alpha=0.8
        )

    axes[0, 1].set_title('Avg Queue Length per Junction', fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Vehicles')  # number of cars
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)


    # Graph 3: signal timer duration
    for i in range(2):
        axes[0, 2].plot(
            smooth(junction_timers[i]),  # green light duration
            color=colors[i],
            label=f'Junction {i}',
            alpha=0.8
        )

    axes[0, 2].set_title('Mean Signal Duration (Timer)', fontweight='bold')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Seconds')  # time in seconds
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)


    # ================= ROW 1 =================
    # System-wide metrics

    # Graph 4: total system reward
    axes[1, 0].plot(
        smooth(system_rewards),   # smoothed system reward
        color='#378ADD',
        linewidth=2
    )

    axes[1, 0].set_title('System Aggregate Reward', fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Sum of Rewards')
    axes[1, 0].grid(True, alpha=0.3)


    # Graph 5: total system queue
    axes[1, 1].plot(
        smooth(system_queues),   # total vehicles waiting
        color='#BA7517',
        linewidth=2
    )

    axes[1, 1].set_title('Network-wide Total Queue', fontweight='bold')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Total Vehicles Waiting')
    axes[1, 1].grid(True, alpha=0.3)


    # Graph 6: throughput (vehicles served)
    for i in range(2):
        axes[1, 2].plot(
            smooth(junction_served[i]),  # vehicles passed
            color=colors[i],
            label=f'Junction {i}',
            alpha=0.8
        )

    axes[1, 2].set_title('Throughput (Vehicles Served)', fontweight='bold')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Count')  # number of vehicles
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)


    # ================= ROW 2 =================
    # Advanced analysis

    # Graph 7: queue difference between junctions
    q_diff = [
        abs(q0 - q1)   # absolute difference
        for q0, q1 in zip(junction_queues[0], junction_queues[1])
    ]
    # IMPORTANT: measures imbalance

    axes[2, 0].plot(
        smooth(q_diff),
        color='#9B59B6',
        linewidth=2
    )

    axes[2, 0].set_title('Traffic Load Imbalance (|J0 - J1|)', fontweight='bold')
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Vehicle Difference')
    axes[2, 0].grid(True, alpha=0.3)


    # Graph 8: reward variance (stability)
    j0_var = [
        np.var(junction_rewards[0][max(0, k-10):k+1])
        # variance of last 10 values
        for k in range(len(junction_rewards[0]))
    ]
    # IMPORTANT: shows stability

    axes[2, 1].plot(
        smooth(j0_var),
        color='#E67E22',
        linewidth=2
    )

    axes[2, 1].set_title('Training Stability (Reward Var)', fontweight='bold')
    axes[2, 1].set_xlabel('Episode')
    axes[2, 1].set_ylabel('Variance')
    axes[2, 1].grid(True, alpha=0.3)


    # Graph 9: long-term learning progress
    window = 50  # number of episodes for average

    for i in range(2):
        ma = [
            np.mean(junction_queues[i][max(0, k-window):k+1])
            # moving average
            for k in range(len(junction_queues[i]))
        ]

        axes[2, 2].plot(
            ma,
            color=colors[i],
            label=f'Junction {i}',
            linewidth=2
        )

    axes[2, 2].set_title(f'Long-term Progress ({window}-ep MA)', fontweight='bold')
    axes[2, 2].set_xlabel('Episode')
    axes[2, 2].set_ylabel('Avg Queue')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)


    # adjust layout so graphs don’t overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # IMPORTANT: formatting

    # save image to file
    plt.savefig(filename)
    # IMPORTANT: saves output

    # close plot to free memory
    plt.close()

    # print confirmation
    print(f"9-Graph Dashboard saved to {filename}")


# function to print training progress
def print_metrics(episode, reward, queue, agents):

    print(
        f"Ep {episode:4d} | "          # episode number
        f"Total Reward: {reward:8.1f} | "  # reward
        f"Avg Queue: {queue:5.1f} | "      # queue
        f"eps: {agents[0].epsilon:.3f}"    # exploration rate
    )
    # IMPORTANT: used to monitor training
