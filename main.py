import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless mode

from src.env import MultiJunctionSystem
from src.agent import MultiAgentQLearning
from src.utils import plot_performance, print_metrics

def main():
    # Configuration
    N_EPISODES = 500
    STEPS_PER_EP = 100
    ARRIVAL_RATE = 2.0
    
    # Initialize Environment and Agents
    system = MultiJunctionSystem(n_junctions=2, arrival_rate=ARRIVAL_RATE)
    agents = [MultiAgentQLearning(eps_decay_rate=0.99) for _ in range(2)]
    
    # System-wide history
    history_rewards = []
    history_queues = []
    
    # Per-junction history
    history_j_rewards = [[], []]
    history_j_queues = [[], []]
    history_j_timers = [[], []]
    history_j_served = [[], []]
    
    print("=" * 60)
    print("  Starting Multi-Agent Traffic Simulation Training")
    print("=" * 60)
    print(f"  Topology: 2 Junctions (Coordinated)")
    print(f"  Episodes: {N_EPISODES}")
    print(f"  Learning: Optimized State Space + Per-Episode Decay")
    print("=" * 60)
    
    for ep in range(1, N_EPISODES + 1):
        states = system.reset()
        ep_reward = 0
        ep_queue = 0
        
        # Track per-junction metrics for this episode
        j_ep_rewards = [0, 0]
        j_ep_queues = [0, 0]
        j_ep_timers = [0, 0]
        j_ep_served = [0, 0]
        
        for step in range(STEPS_PER_EP):
            actions = [agent.choose_action(s) for agent, s in zip(agents, states)]
            next_states, rewards, infos = system.step(actions)
            
            for i in range(2):
                agents[i].update(states[i], actions[i], rewards[i], next_states[i])
                j_ep_rewards[i] += rewards[i]
                j_ep_queues[i] += sum(system.junctions[i].queues.values())
                j_ep_timers[i] += infos[i]['green_time']
                j_ep_served[i] += infos[i]['served']
                
            ep_reward += sum(rewards)
            ep_queue += sum(sum(j.queues.values()) for j in system.junctions)
            states = next_states
            
        # Per-episode cleanup and history update
        for i in range(2):
            agents[i].decay_epsilon()
            history_j_rewards[i].append(j_ep_rewards[i])
            history_j_queues[i].append(j_ep_queues[i] / STEPS_PER_EP)
            history_j_timers[i].append(j_ep_timers[i] / STEPS_PER_EP)
            history_j_served[i].append(j_ep_served[i])
            
        history_rewards.append(ep_reward)
        history_queues.append(ep_queue / (STEPS_PER_EP * 2))
        
        if ep % 50 == 0:
            avg_reward = np.mean(history_rewards[-50:])
            avg_queue = np.mean(history_queues[-50:])
            print_metrics(ep, avg_reward, avg_queue, agents)
            
    print("\nTraining complete!")
    
    # Final Evaluation
    print("\nRunning evaluation (eps=0)...")
    eval_queues = []
    for _ in range(20):
        states = system.reset()
        ep_q = 0
        for _ in range(STEPS_PER_EP):
            actions = [int(np.argmax(agent.get_q_values(s))) for agent, s in zip(agents, states)]
            states, _, _ = system.step(actions)
            ep_q += sum(sum(j.queues.values()) for j in system.junctions)
        eval_queues.append(ep_q / (STEPS_PER_EP * 2))
        
    print(f"Mean Evaluation Queue: {np.mean(eval_queues):.2f} vehicles")
    
    # Generate Plots (now 9 graphs)
    plot_performance(
        history_rewards, 
        history_queues, 
        history_j_rewards, 
        history_j_queues,
        history_j_timers,
        history_j_served,
        'optimized_traffic_results.png'
    )
    
if __name__ == "__main__":
    main()
