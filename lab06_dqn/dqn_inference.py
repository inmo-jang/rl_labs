import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import torch
from gyms.simple_maze_grid import SimpleMazeGrid

from lab06_dqn.dqn import DQN, state_to_tensor, load_model, select_action
# from lab06_dqn.dqn_solution import DQN, state_to_tensor, load_model, select_action
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)


def play_inference(env, policy_net, device, num_episodes=5, random_seed_list = None):
    total_rewards = []
    for i_episode in range(num_episodes):
        random_seed = None
        if random_seed_list is not None:
            random_seed = random_seed_list[i_episode]
            
        state, _ = env.reset(random_seed)
            
        player_pos = env.get_player_pos()
        state_tensor = state_to_tensor(state, player_pos).to(device)
        
        terminated = False
        total_reward = 0

        while not terminated:
            action = select_action(state_tensor, policy_net, device, epsilon=0.0, n_actions=env.action_space.n)  # epsilon=0.0 for greedy policy
            next_state, reward, terminated, _ = env.step(action.item())
            total_reward += reward
            
            if terminated or total_reward <= -100:
                print(f"Episode {i_episode + 1}: Total Reward: {total_reward}")
                break
            
            next_player_pos = env.get_player_pos()
            state_tensor = state_to_tensor(next_state, next_player_pos).to(device)
            
            env.render(fps = 5)
        
        total_rewards.append(total_reward)
    return np.average(total_rewards)

def main():
    # Initialize environment
    n = 5  # Grid size
    k = 3  # Minimum goal position
    m = 4  # Number of pits    
    render_option = True  # Set to True for rendering during inference
    
    random_seed_list = [25,32,123,56,34,71,93,82,19,1320]

    env = SimpleMazeGrid(n=n, k=k, m=m, render_option=render_option)
    state, _ = env.reset()
    player_pos = env.get_player_pos()
    state_tensor = state_to_tensor(state, player_pos).to(device)        
    input_shape = state_tensor.squeeze(0).shape
    policy_net = DQN(input_shape, env.action_space.n).to(device)

    # Load the trained model
    model_path = 'policy_net_300.pth'  # Change to the path of your saved model
    load_model(policy_net, model_path)

    # Play inference
    average_reward = play_inference(env, policy_net, device, num_episodes=10, random_seed_list=random_seed_list)
    print(f"average_reward = {average_reward}")

    env.close()

if __name__ == "__main__":
    main()
