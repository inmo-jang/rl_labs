import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
from gyms.simple_maze_grid import SimpleMazeGrid

def state_to_index(state, n):
    player_pos = np.argwhere(state == 1)
    if player_pos.size == 0:
        return None
    row, col = player_pos[0]
    return row * n + col

def policy_evaluation(env, policy, gamma=0.99, theta=1e-6):
    V = np.zeros(env.n * env.n)
    delta = float('inf')
    
    while delta > theta:
        delta = 0
        for player_pos in env.get_all_states():
            env.set_player_pos(player_pos)
            state = env._get_state()
            state_idx = state_to_index(state, env.n)
            if state_idx is None:
                continue
            v = V[state_idx]

            # TODO: Implement the policy evaluation algorithm


        # NOTE: The following can be used to visualise this algorithm's process
        # if env.render_option:
        #     env.render_v_values(V, policy, 'Eval.')    

    return V

def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros((env.n * env.n, env.action_space.n))
    for player_pos in env.get_all_states():
        env.set_player_pos(player_pos)
        state = env._get_state()
        state_idx = state_to_index(state, env.n)
        if state_idx is None:
            continue        
        # TODO: Implement the policy improvement algorithm        
    
        # NOTE: The following can be used to visualise this algorithm's process
        # if env.render_option:
        #     env.render_v_values(V, policy, 'Imp.')  

    return policy

def policy_iteration(env, gamma=0.99, theta=1e-6):
    policy = np.ones((env.n * env.n, env.action_space.n)) / env.action_space.n  # Initial policy (uniform random)
    iteration = 0
    
    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)
        
        if np.all(policy == new_policy):
            break
        
        policy = new_policy
        iteration += 1        
    
    return policy, V, iteration

def main():
    n = 5
    k = 3
    m = 4
    random_seed = 42
    gamma = 0.9
    theta = 1e-6

    env = SimpleMazeGrid(n=n, k=k, m=m, render_option=True, random_seed=random_seed)
    env.render()

    policy, V, iteration = policy_iteration(env, gamma=gamma, theta=theta)

    # Plot the value function
    env.render_v_values(V, policy, iteration)
    plt.imshow(V.reshape((n, n)), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Value Function')
    plt.savefig('policy_iteration_simple_maze_grid.png')
    plt.show()

    print("Completed!")

if __name__ == "__main__":
    main()
