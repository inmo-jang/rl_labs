import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
from gyms.simple_maze_grid import SimpleMazeGrid

# Define softmax function for policy
def softmax(x):
    # e_x = np.exp(x - np.max(x))
    e_x = np.exp(x)
    return e_x / e_x.sum()

# Convert state to index
def state_to_index(state, n):
    player_pos = np.argwhere(state == 1)
    if player_pos.size == 0:
        return None
    row, col = player_pos[0]
    return row * n + col

# Sample action based on softmax policy
def choose_action(state_index, policy_params):
    action_probs = softmax(policy_params[state_index])
    return np.random.choice(len(action_probs), p=action_probs)


def generate_episode(env, policy_params, trajectory_length, episode_index):
    # Reset environment
    state, _ = env.retry()

    state_index = state_to_index(state, env.n)
    
    # Generate an episode
    trajectory = []
    total_reward = 0

    # Run episode and store transitions
    for _ in range(trajectory_length):  # Limiting the trajectory length
        if state_index is None:  # The player is at either pit or goal. 
            break        

        # Record the state-action pair and its reward
        action = choose_action(state_index, policy_params)
        next_state, reward, terminated, _ = env.step(action)        
        trajectory.append((state_index, action, reward))

        if env.render_option:
            env.render_q_values(policy_params, episode_index, with_arrow=True)

        next_state_index = state_to_index(next_state, env.n)
        state_index = next_state_index
        total_reward += reward

    return trajectory, total_reward

def reinforce_learning(env, num_episodes=100, alpha=0.01, gamma=0.99, trajectory_length = 300, render_option=False):
    policy_params = np.random.rand(env.n * env.n, env.action_space.n)
    total_rewards = []

    for episode_index in range(num_episodes):
        # Run episode and store transitions
        trajectory, total_reward = generate_episode(env, policy_params, trajectory_length, episode_index)

        # Compute returns and update policy after episode ends
        G = 0
        for t in reversed(range(len(trajectory))):
            state_index, action, reward = trajectory[t]
            G = gamma * G + reward
            action_probs = softmax(policy_params[state_index])

            # Update policy parameters following the REINFORCE update rule            
            grad_log_pi = np.zeros_like(action_probs)
            grad_log_pi[action] = 1 - action_probs[action]  # For chosen action a
            policy_params[state_index] += alpha * gamma**t * G * grad_log_pi            

        total_rewards.append(total_reward)

    return policy_params, total_rewards

def main():
    num_episodes = 1000
    alpha = 0.01  # Learning rate
    gamma = 0.9  # Discount factor

    n = 5  # Grid size
    render_option = True
    trajectory_length = n * 50

    # Initialize environment
    player_pos = [n-1, 0]
    goal_pos = [n-1, n-1]
    pits = [[n-1, i] for i in range(1, n-1)]
    spec = player_pos, goal_pos, pits
    env = SimpleMazeGrid(n=n, render_option=render_option, spec=spec)

    policy_params, total_rewards = reinforce_learning(env, num_episodes=num_episodes, alpha=alpha, gamma=gamma, trajectory_length=trajectory_length, render_option=render_option)

    # Plot the rewards
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE Learning Training')
    plt.grid(True)
    plt.savefig('reinforce_simple_maze_grid.png')
    plt.show()

    print("Completed!")

if __name__ == "__main__":
    main()
