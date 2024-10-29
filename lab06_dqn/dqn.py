import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random
import torch.nn.functional as F


from gyms.simple_maze_grid import SimpleMazeGrid


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Define a named tuple for storing experiences
Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward'])

# Replay Memory Class to store and sample experiences
class ReplayMemory(object): # Exactly the same as https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Randomly samples batch_size transitions from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the current size of internal memory."""
        return len(self.memory)

    
# TODO: (1) Design your DQN model
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        input_size = np.prod(input_shape)
        self.fc1 = nn.Linear(input_size, 256)  
        self.fc2 = nn.Linear(256, 512)         
        self.fc3 = nn.Linear(512, 512)         
        self.fc4 = nn.Linear(512, n_actions)      

    def forward(self, x):
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    
  

# TODO: (2) Interfacing observation to the DQN model
def state_to_tensor(state, player_pos):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
    return state_tensor



def select_action(state_tensor, policy_net, device, epsilon, n_actions): # Almost the same as https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    if np.random.rand() > epsilon:
        with torch.no_grad():
            return policy_net(state_tensor).max(1).indices.view(1,1)
        
    else:                   
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
   
def optimize_model(memory, policy_net, target_net, optimizer, device, BATCH_SIZE = 128, GAMMA = 0.99): # Exactly the same as https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
def dqn_learning(env, policy_net, target_net, memory, num_episodes=100, alpha=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99, batch_size=64, tau=0.001, render_option=False, start_episode=0, print_interval = 20, model_save_interval = 500):
    optimizer = optim.Adam(policy_net.parameters(), lr=alpha, amsgrad=True)
    total_rewards = []

    epsilon = epsilon_start
    for i_episode in range(num_episodes):
        state, _ = env.reset()
        player_pos = env.get_player_pos()
        state_tensor = state_to_tensor(state, player_pos).to(device)
        
        terminated = False
        score = 0.0
        while not terminated:
            action = select_action(state_tensor, policy_net, device, epsilon, env.action_space.n)
            next_state, reward, terminated, _ = env.step(action.item())                
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            next_player_pos = env.get_player_pos()
            
            if terminated:
                next_state_tensor = None
            else:
                next_state_tensor = state_to_tensor(next_state, next_player_pos).to(device)
                

            # Store the transition in memory    
            memory.push(state_tensor, action, next_state_tensor, reward)            
            
            # Move to the next state
            state_tensor = next_state_tensor
            
            # Perform one step of the opimization (on the policy network)
            optimize_model(memory, policy_net, target_net, optimizer, device, batch_size, gamma)

            # Soft update of the target network's weights: θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            target_net.load_state_dict(target_net_state_dict)
                            
            score += reward  
            
                          
            if render_option:
                env.render()


                    
        total_rewards.append(score.item())
        
        if i_episode % print_interval==0 and i_episode > 0:    
            print(f"n_episode :{i_episode + start_episode}, score : {sum(total_rewards[-print_interval:])/print_interval:.2f}, n_buffer : {memory.__len__()}, eps : {epsilon:.3f}")                    


        if i_episode % model_save_interval == 0:
            save_model(policy_net, i_episode + start_episode)        
        
        # Update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)        



    return total_rewards


def save_model(model, episode):
    """Saves the model state_dict."""
    torch.save(model.state_dict(), f'policy_net_{episode}.pth')

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()


def main():
    num_runs = 1
    render_option = False

    # TODO: (3) Set your parameters
    num_episodes = 10000
    alpha = 1e-4  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon_start = 0.9  # Initial epsilon for exploration
    epsilon_end = 0.05    # Final epsilon for exploitation
    epsilon_decay = 0.99  # Decay factor for epsilon
    replay_memory_size = 10000
    batch_size = 128
    tau = 0.005  # Soft update parameter

    # Initialize environment
    n = 5  # Grid size
    k = 3  # Minimum goal position
    m = 4  # Number of pits
    random_seed = None
    
    # New variables for resuming training
    start_episode = 0
    load_model_path = None  # Set this to the path of the model you want to load, e.g., 'policy_net_9500.pth'

    print_interval = 20
    model_save_interval = 1000
        
    all_rewards = np.zeros((num_runs, num_episodes))

    for run in range(num_runs):
        env = SimpleMazeGrid(n=n, k=k, m=m, render_option=render_option, random_seed=random_seed)
        # Initialise replay memory
        memory = ReplayMemory(replay_memory_size)
        # Initialise action-value function Q
        state, _ = env.retry()
        player_pos = env.get_player_pos()
        state_tensor = state_to_tensor(state, player_pos).to(device)        
        input_shape = state_tensor.squeeze(0).shape
        policy_net = DQN(input_shape, env.action_space.n).to(device)
        # Initialise target action-value function Q_hat
        target_net = DQN(input_shape, env.action_space.n).to(device)
                
        # Load model (if path is specified) to the policy network
        if load_model_path:
            load_model(policy_net, load_model_path)
            start_episode = int(load_model_path.split('_')[-1].split('.')[0]) + 1  # Assuming the file name format is 'policy_net_<episode>.pth'
            print(f"Success loading the existing model: {load_model_path}")
        # Clone the policy network's weight to the target network
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval() # Set as Evaluation mode (Not for training)

        total_rewards = dqn_learning(env, policy_net, target_net, memory, num_episodes=num_episodes, alpha=alpha, gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay=epsilon_decay, batch_size=batch_size, tau=tau, render_option=render_option, start_episode = start_episode, print_interval=print_interval, model_save_interval=model_save_interval)
        all_rewards[run] = total_rewards
        env.close()

    average_rewards = np.mean(all_rewards, axis=0)

    # Plot the average rewards over all runs
    plt.plot(average_rewards, label=f'$\\alpha$ = {alpha}; $\\epsilon_{{start}}$ = {epsilon_start}; $\\epsilon_{{end}}$ = {epsilon_end}')
    plt.xlabel('Episode')
    plt.ylabel('Average Total Reward')
    plt.title(f'DQN Training (Average Over {num_runs} Runs)')
    plt.grid(True)
    plt.legend()
    plt.savefig('dqn_reward.png')
    plt.show()

    print("Completed!")

if __name__ == "__main__":
    main()
