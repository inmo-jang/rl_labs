from baseline.cartpole import CartPoleEnv
from gym.envs.registration import register
import gym
import pygame
import time
import numpy as np
from typing import Optional

class CartPolePendulumEnv(CartPoleEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.theta_init = np.pi
        self.theta_threshold_radians = float('inf')        

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.state, info = super().reset(seed=seed, options=options) # 기존 CartPole의 reset을 호출하여 초기화
        self.state[2] = self.theta_init # 초기 추의 위치를 theta_init 값으로 설정
        return self.state, {}


class PendulumGym:
    def __init__(self, render_mode=None, episode_length = 1000):
        register(
            id='CartPolePendulumEnv',  
            entry_point='__main__:CartPolePendulumEnv',  
        )
        self.env = gym.make('CartPolePendulumEnv', render_mode=render_mode)
        self.env.reset()
        
        # For video recording
        self.render_mode = render_mode
        self.metadata = self.env.metadata
        
        # For truncation
        self.steps = 0 
        self.episode_length = episode_length

    def get_env(self):
        return self.env
    
    def render(self):
        return self.env.render()
    
    def reset(self, seed=None, options=None):
        self.steps = 0
        return self.env.reset(seed=seed, options=options)
    
    def close(self):
        self.env.close()
    
    def step(self, action):
        next_state, reward, terminated, truncated, _  = self.env.step(action)   
                  
        # theta range [-2pi ~ 2pi]
        next_state[2] = self.normalize_angle(next_state[2])

        # Truncation check
        self.steps += 1
        if self.steps > self.episode_length:
            truncated = True
                 
        return next_state, reward, terminated, truncated, _ 

    def get_action_space(self):
        return self.env.action_space
    
    def get_observation_space(self):
        return self.env.observation_space
    
    def normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))    

    def key_action(self, previous_action = 0):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            return 0  # Move left
        elif keys[pygame.K_RIGHT]:
            return 1  # Move right
        elif keys[pygame.K_r]:
            self.reset()

        return previous_action  # Default action
        
    def handle_keyboard_input(self):
        clock = pygame.time.Clock()
        terminated = False    
        previous_action = 0
        while not terminated:
            self.env.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            action = self.key_action(previous_action)        
            next_state, reward, terminated, truncated, _  = self.step(action)            
            print(f"reward = {reward:.1f}, pos = {next_state[0]:.2f}, vel = {next_state[1]:.2f}, theta = {next_state[2]:.2f}, theta_dot = {next_state[3]:.2f}")
            pygame.display.flip()
            clock.tick(30)  # FPS
            previous_action = action

        pygame.quit()
        self.env.close()



if __name__ == "__main__":

    # Example usage
    env = PendulumGym(render_mode="human")
    env.reset()
    env.handle_keyboard_input()