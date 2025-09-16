import numpy as np
import random
import pygame
from gym import spaces
import sys
import argparse
import os

class SimpleMazeGrid:
    def __init__(self, n, k=None, m=None, render_option=False, random_seed=None, stochastic=False, epsilon=0.1, spec=None, reward=10):
        self.n = n
        self.k = k
        self.m = m
        self.render_option = render_option
        self.terminated = False
        self.stochastic = stochastic  # Stochastic option
        self.epsilon = epsilon  # Epsilon for stochastic behaviour        
        self.reward = reward
        if random_seed is None and spec is not None:
            initial_player_pos, goal_pos, pits  = spec
            self.initial_player_pos = initial_player_pos # [x,y]
            self.player_pos = [self.initial_player_pos[0], self.initial_player_pos[1]]
            self.goal_pos = goal_pos # [x,y]
            self.pits = pits # [[a,b], [c,d], ...]
            self.cumulative_reward = 0
            self.steps = 0            
        
        elif random_seed is not None:
            self.reset(random_seed)
        
        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.observation_space = spaces.Discrete(n * n)
        
        if self.render_option:
            pygame.init()
            self.screen_width = 1000
            self.screen_height = 1000
            self.info_width = 200
            self.total_width = self.screen_width + self.info_width
            self.screen = pygame.display.set_mode((self.total_width, self.screen_height))
            pygame.display.set_caption("Simple Maze Grid Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
                            # Draw the Q-values as text in the grid cells
            self.small_font = pygame.font.Font(None, 20)  # Smaller font size for Q-values
            # --- Sprite for the agent ---
            self.assets_dir = os.path.join(os.path.dirname(__file__), "assets")
            self.agent_img = None
            self.agent_img_scaled = None  # cache for a given cell_size

            agent_path = os.path.join(self.assets_dir, "drone_sideview.png")
            try:
                self.agent_img = pygame.image.load(agent_path).convert_alpha()
            except Exception as e:
                print(f"[WARN] Failed to load agent image at {agent_path}: {e}. Falling back to circle.")
                        
        
    def reset(self, random_seed = None):

        self.initial_player_pos = [0, 0]
        self.player_pos = self.initial_player_pos
        self.terminated = False  # Reset termination status

        # Set the random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)

        # Initialize positions
        self.goal_pos = [random.randint(self.k, self.n-1), random.randint(self.k, self.n-1)]
        self.pits = []
        while len(self.pits) < self.m:
            pit = [random.randint(0, self.n-1), random.randint(0, self.n-1)]
            if pit != [0, 0] and pit != self.goal_pos and pit not in self.pits:
                self.pits.append(pit)
        
        self.cumulative_reward = 0
        self.steps = 0
        
        return self._get_state(), {}


    def retry(self):
        self.player_pos = [self.initial_player_pos[0], self.initial_player_pos[1]]
        self.terminated = False  # Reset termination status
        self.cumulative_reward = 0
        self.steps = 0
        
        return self._get_state(), {}


    def step(self, action):
        if self.terminated:  # Prevent action if game is terminated
            return self._get_state(), 0, True, {}
        
        # Stochastic action selection
        if self.stochastic and random.random() < self.epsilon:
            action = self.action_space.sample()
        

        if action == 0:  # Up
            self.player_pos[0] = max(0, self.player_pos[0] - 1)
        elif action == 1:  # Down
            self.player_pos[0] = min(self.n - 1, self.player_pos[0] + 1)
        elif action == 2:  # Left
            self.player_pos[1] = max(0, self.player_pos[1] - 1)
        elif action == 3:  # Right
            self.player_pos[1] = min(self.n - 1, self.player_pos[1] + 1)
        
        reward = -1
        self.terminated = False
        if self.player_pos == self.goal_pos:
            reward = self.reward
            self.terminated = True
        elif self.player_pos in self.pits:
            reward = -self.reward
            self.terminated = True
            
        self.cumulative_reward += reward
        self.steps += 1
            
        
        return self._get_state(), reward, self.terminated, {}
    
    def _get_state(self):
        state = np.zeros((self.n, self.n))
        state[self.player_pos[0], self.player_pos[1]] = 1
        state[self.goal_pos[0], self.goal_pos[1]] = 2
        for pit in self.pits:
            state[pit[0], pit[1]] = 3
        return state

    def _get_scaled_agent(self, cell_size):
        """Return a cached, smoothly scaled sprite that fits the current cell size."""
        if self.agent_img is None:
            return None
        target = int(cell_size * 0.8)  # 셀 크기의 80%로 살짝 여유 있게
        # 정사각형 스케일 (원본이 직사각형이면 비율 유지하고 싶은 경우 fit 로직으로 바꿔도 됨)
        if (self.agent_img_scaled is None or
            self.agent_img_scaled.get_width() != target or
            self.agent_img_scaled.get_height() != target):
            self.agent_img_scaled = pygame.transform.smoothscale(self.agent_img, (target, target))
        return self.agent_img_scaled

    def _blit_agent(self, row, col, cell_size):
        """Center the agent sprite in the (row, col) cell. Fallback to circle if sprite missing."""
        cx = col * cell_size + cell_size // 2
        cy = row * cell_size + cell_size // 2
        sprite = self._get_scaled_agent(cell_size)
        if sprite is not None:
            rect = sprite.get_rect(center=(cx, cy))
            self.screen.blit(sprite, rect)
        else:
            pygame.draw.circle(self.screen, (0, 0, 255), (cx, cy), cell_size // 3)

    def render(self, fps = 30):
        self.screen.fill((255, 255, 255))
        cell_size = self.screen_width // self.n
        
        for i in range(self.n):
            for j in range(self.n):
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
        
        pygame.draw.rect(self.screen, (0, 255, 0), (self.goal_pos[1] * cell_size, self.goal_pos[0] * cell_size, cell_size, cell_size))
        
        for pit in self.pits:
            pygame.draw.rect(self.screen, (255, 0, 0), (pit[1] * cell_size, pit[0] * cell_size, cell_size, cell_size))
        
        # pygame.draw.circle(self.screen, (0, 0, 255), (self.player_pos[1] * cell_size + cell_size // 2, self.player_pos[0] * cell_size + cell_size // 2), cell_size // 3)
        # Draw the agent sprite or fallback to circle
        self._blit_agent(self.player_pos[0], self.player_pos[1], cell_size)      

        # Draw score board
        pygame.draw.rect(self.screen, (255, 255, 255), (self.screen_width, 0, self.info_width, self.screen_height))
        score_text = self.font.render(f"Return: {self.cumulative_reward}", True, (0, 0, 0))
        steps_text = self.font.render(f"Steps: {self.steps}", True, (0, 0, 0))
        self.screen.blit(score_text, (self.screen_width + 10, 10))
        self.screen.blit(steps_text, (self.screen_width + 10, 50))

        if self.terminated:
            finished_text = self.font.render("FINISHED", True, (0, 0, 0))
            self.screen.blit(finished_text, (self.screen_width // 2 - 50, self.screen_height // 2 - 20))
        pygame.display.flip()
        self.clock.tick(fps)
                

        
         
    def render_q_values(self, q_table, episode_number, with_arrow = False):

        pygame.event.get()
 

        self.screen.fill((255, 255, 255))
        cell_size = self.screen_width // self.n
        
        for i in range(self.n):
            for j in range(self.n):
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                
                # Get the Q-values for the current cell
                q_values = q_table[i * self.n + j]
                
                SENSITIVITY = 3.0
                # Draw diagonals with color corresponding to Q-values
                for action in range(4):
                    q_a_value = q_values[action]
                    if q_a_value <= 0:  # Q-value is negative
                        red_color_intensity = int(255 + q_a_value * SENSITIVITY)
                        red_color_intensity = min(max(red_color_intensity, 0), 255)  # Clamp to [0, 255]
                        color_code = (255, red_color_intensity, red_color_intensity)
                    else:  # Q-value is positive or zero
                        green_color_intensity = int(255 - q_a_value * SENSITIVITY)
                        green_color_intensity = min(max(green_color_intensity, 0), 255)  # Clamp to [0, 255]
                        color_code = (green_color_intensity, 255, green_color_intensity)



                    if action == 0:  # Up
                        pygame.draw.polygon(self.screen, color_code, [
                            (j * cell_size, i * cell_size),
                            ((j + 1) * cell_size, i * cell_size),
                            (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2)
                        ])
                    elif action == 1:  # Down
                        pygame.draw.polygon(self.screen, color_code, [
                            (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2),
                            ((j + 1) * cell_size, (i + 1) * cell_size),
                            (j * cell_size, (i + 1) * cell_size)
                        ])
                    elif action == 2:  # Left
                        pygame.draw.polygon(self.screen, color_code, [
                            (j * cell_size, i * cell_size),
                            (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2),
                            (j * cell_size, (i + 1) * cell_size)
                        ])
                    elif action == 3:  # Right
                        pygame.draw.polygon(self.screen, color_code, [
                            (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2),
                            ((j + 1) * cell_size, i * cell_size),
                            ((j + 1) * cell_size, (i + 1) * cell_size)
                        ])



                # Draw Q-values
                for action in range(4):
                    text_surface = self.small_font.render(f"{q_values[action]:.1f}", True, (0, 0, 0))
                    # Positioning text to avoid overlap and to fit well in the cell
                    if action == 0:  # Up
                        text_x = j * cell_size + 10 +(cell_size // 3)
                        text_y = i * cell_size + 2 
                    elif action == 1:  # Down
                        text_x = j * cell_size + 10 + 1 * (cell_size // 3)
                        text_y = i * cell_size + 2 + 2 * (cell_size // 3)
                    elif action == 2:  # Left
                        text_x = j * cell_size + 10
                        text_y = i * cell_size + 2 + 1 * (cell_size // 3)
                    elif action == 3:  # Right
                        text_x = j * cell_size + 10 + 2 * (cell_size // 3)
                        text_y = i * cell_size + 2 + 1 * (cell_size // 3)

                    self.screen.blit(text_surface, (text_x, text_y))

                # Draw each grid
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

                if with_arrow:
                    # 정책에 따른 화살표 그리기
                    action = np.argmax(q_values)
                    # 방향 화살표 좌표 설정
                    arrow_offset = int(cell_size / 4)
                    arrow_size = int(cell_size / 4)                                    

                    # 화살표 방향 설정
                    if action == 0:  # up
                        arrow_points = [(j * cell_size + cell_size // 2, i * cell_size + arrow_offset),
                                        (j * cell_size + arrow_offset, i * cell_size + arrow_offset + arrow_size),
                                        (j * cell_size + cell_size - arrow_offset, i * cell_size + arrow_offset + arrow_size)]
                    elif action == 1:  # down
                        arrow_points = [(j * cell_size + cell_size // 2, i * cell_size + cell_size - arrow_offset),
                                        (j * cell_size + arrow_offset, i * cell_size + cell_size - arrow_offset - arrow_size),
                                        (j * cell_size + cell_size - arrow_offset, i * cell_size + cell_size - arrow_offset - arrow_size)]
                    elif action == 2:  # left
                        arrow_points = [(j * cell_size + arrow_offset, i * cell_size + cell_size // 2),
                                        (j * cell_size + arrow_offset + arrow_size, i * cell_size + arrow_offset),
                                        (j * cell_size + arrow_offset + arrow_size, i * cell_size + cell_size - arrow_offset)]
                    elif action == 3:  # right
                        arrow_points = [(j * cell_size + cell_size - arrow_offset, i * cell_size + cell_size // 2),
                                        (j * cell_size + cell_size - arrow_offset - arrow_size, i * cell_size + arrow_offset),
                                        (j * cell_size + cell_size - arrow_offset - arrow_size, i * cell_size + cell_size - arrow_offset)]

                    pygame.draw.polygon(self.screen, (120, 120, 255), arrow_points)  # 화살표 그리기  





        
        # Draw the goal
        pygame.draw.rect(self.screen, (0, 255, 0), (self.goal_pos[1] * cell_size, self.goal_pos[0] * cell_size, cell_size, cell_size))
        
        # Draw the pits
        for pit in self.pits:
            pygame.draw.rect(self.screen, (255, 0, 0), (pit[1] * cell_size, pit[0] * cell_size, cell_size, cell_size))
        
        # Draw the player
        # pygame.draw.circle(self.screen, (0, 0, 255), (self.player_pos[1] * cell_size + cell_size // 2, self.player_pos[0] * cell_size + cell_size // 2), cell_size // 3)
        self._blit_agent(self.player_pos[0], self.player_pos[1], cell_size)

        # Draw score board
        pygame.draw.rect(self.screen, (255, 255, 255), (self.screen_width, 0, self.info_width, self.screen_height))
        episode_number_text = self.font.render(f"Episode #: {episode_number}", True, (0, 0, 0))
        score_text = self.font.render(f"Return: {self.cumulative_reward}", True, (0, 0, 0))
        steps_text = self.font.render(f"Steps: {self.steps}", True, (0, 0, 0))
        self.screen.blit(episode_number_text, (self.screen_width + 10, 10))
        self.screen.blit(score_text, (self.screen_width + 10, 50))
        self.screen.blit(steps_text, (self.screen_width + 10, 90))
        
        
        pygame.display.flip()        
        self.clock.tick(30)


    def close(self):
        if self.render_option:
            pygame.quit()
            sys.exit()

    def handle_keyboard_input(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and not self.terminated:                        
                        next_state, reward, terminated, _ = self.step(0)
                    elif event.key == pygame.K_DOWN and not self.terminated:
                        next_state, reward, terminated, _ = self.step(1)
                    elif event.key == pygame.K_LEFT and not self.terminated:
                        next_state, reward, terminated, _ = self.step(2)
                    elif event.key == pygame.K_RIGHT and not self.terminated:
                        next_state, reward, terminated, _ = self.step(3)
                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE: # Quit
                        running = False
                    elif event.key == pygame.K_r: # Reset
                        next_state, reward = self.reset()
                    else:
                        next_state = {}
                        reward = {}

                    if self.render_option:
                        self.render()
                        print(f"state = \n{next_state}; \nreward = {reward}")
                    
        
        self.close()


    def set_player_pos(self, player_pos):
        self.player_pos = player_pos[:]

    def get_player_pos(self):
        return self.player_pos[:]

    def get_all_states(self):
        all_states = []
        for i in range(self.n):
            for j in range(self.n):
                all_states.append([i, j])
        return all_states

    def simulate_action(self, player_pos, action):
        self.retry()
        self.set_player_pos(player_pos)
        next_state, reward, terminated, _ = self.step(action)
        return next_state, reward, terminated

              
    def render_v_values(self, value_table, policy, iteration_number):
        pygame.event.get()
        self.screen.fill((255, 255, 255))
        cell_size = int(self.screen_width / self.n)
        SENSITIVITY = 10.0

        # 방향 화살표 좌표 설정
        arrow_offset = int(cell_size / 4)
        arrow_size = int(cell_size / 4)        
        
        for i in range(self.n):
            for j in range(self.n):
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                
                value = value_table[i * self.n + j]
                color_intensity = int(255 - (value * SENSITIVITY) if value >= 0 else 255 + (value * SENSITIVITY))
                color_intensity = min(max(color_intensity, 0), 255)
                color = (color_intensity, 255, color_intensity) if value >= 0 else (255, color_intensity, color_intensity)
                
                pygame.draw.rect(self.screen, color, rect)
                text_surface = self.font.render(f"{value:.2f}", True, (0, 0, 0))
                self.screen.blit(text_surface, (j * cell_size + 10, i * cell_size + 10))
                
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

                # 정책에 따른 화살표 그리기
                state_idx = i * self.n + j
                if np.max(policy[state_idx]) > 0:  # 유효한 정책이 있을 때만 그리기
                    action = np.argmax(policy[state_idx])

                    # 화살표 방향 설정
                    if action == 0:  # up
                        arrow_points = [(j * cell_size + cell_size // 2, i * cell_size + arrow_offset),
                                        (j * cell_size + arrow_offset, i * cell_size + arrow_offset + arrow_size),
                                        (j * cell_size + cell_size - arrow_offset, i * cell_size + arrow_offset + arrow_size)]
                    elif action == 1:  # down
                        arrow_points = [(j * cell_size + cell_size // 2, i * cell_size + cell_size - arrow_offset),
                                        (j * cell_size + arrow_offset, i * cell_size + cell_size - arrow_offset - arrow_size),
                                        (j * cell_size + cell_size - arrow_offset, i * cell_size + cell_size - arrow_offset - arrow_size)]
                    elif action == 2:  # left
                        arrow_points = [(j * cell_size + arrow_offset, i * cell_size + cell_size // 2),
                                        (j * cell_size + arrow_offset + arrow_size, i * cell_size + arrow_offset),
                                        (j * cell_size + arrow_offset + arrow_size, i * cell_size + cell_size - arrow_offset)]
                    elif action == 3:  # right
                        arrow_points = [(j * cell_size + cell_size - arrow_offset, i * cell_size + cell_size // 2),
                                        (j * cell_size + cell_size - arrow_offset - arrow_size, i * cell_size + arrow_offset),
                                        (j * cell_size + cell_size - arrow_offset - arrow_size, i * cell_size + cell_size - arrow_offset)]

                    pygame.draw.polygon(self.screen, (120, 120, 255), arrow_points)  # 화살표 그리기                
                
                
        
        pygame.draw.rect(self.screen, (0, 255, 0), (self.goal_pos[1] * cell_size, self.goal_pos[0] * cell_size, cell_size, cell_size))
        for pit in self.pits:
            pygame.draw.rect(self.screen, (255, 0, 0), (pit[1] * cell_size, pit[0] * cell_size, cell_size, cell_size))
        # pygame.draw.circle(self.screen, (0, 0, 255), (self.player_pos[1] * cell_size + cell_size // 2, self.player_pos[0] * cell_size + cell_size // 2), cell_size // 3)
        self._blit_agent(self.player_pos[0], self.player_pos[1], cell_size)


        pygame.draw.rect(self.screen, (255, 255, 255), (self.screen_width, 0, self.info_width, self.screen_height))
        iteration_text = self.font.render(f"Iteration #: {iteration_number}", True, (0, 0, 0))
        self.screen.blit(iteration_text, (self.screen_width + 10, 10))
        
        pygame.display.flip()
        self.clock.tick(30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Maze Grid Environment")
    parser.add_argument('--stochastic', type=bool, default=False, help='Enable stochastic action selection')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Probability of taking a random action when stochastic is enabled')
    args = parser.parse_args()

    # Example usage
    env = SimpleMazeGrid(n=5, k=3, m=2, render_option=True, stochastic=args.stochastic, epsilon=args.epsilon, random_seed=42)
    env.reset()
    env.render()
    env.handle_keyboard_input()
