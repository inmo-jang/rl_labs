import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
current_dir = os.path.dirname(os.path.abspath(__file__))

import torch
from gyms.cartpole_double_pendulum import SerialDoublePendulumGym, CartPoleSerialDoublePendulumEnv
from dqn_utils import DQN
import cv2
import numpy as np

class PerformanceChecker():
    def __init__(self, video_folder_path):
        self.video_folder_path = video_folder_path
        

    def load_model(self, model_path, n_observations, n_actions, device):    
        """Load the saved model from the given path."""
        model = DQN(n_observations, n_actions)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        return model

    def visualize_performance(self, environment, model, device, video_file_name):
        """Record the model's performance in the environment and display cumulative reward."""
        state, info = environment.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False

        # OpenCV Video Writer 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec 사용
        os.makedirs(self.video_folder_path, exist_ok=True)          
        video_path = f"{self.video_folder_path}/{video_file_name}.mp4"
        frame_width, frame_height = environment.render().shape[1], environment.render().shape[0]
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (frame_width, frame_height))

        step = 0
        while not done:
            with torch.no_grad():
                action = model(state).max(1)[1].view(1, 1)        
            next_state, reward, terminated, truncated, _  = environment.step(action.item())
            done = terminated or truncated                    
            state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            # 환경으로부터 현재 프레임을 가져와서 OpenCV를 이용해 텍스트 추가
            frame = environment.render()  # 프레임을 가져오기 (H, W, C) 형식의 배열
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB to BGR 변환 (OpenCV 형식)

      
            # OpenCV로 텍스트 추가
            text = f"Step: {step}"
            cv2.putText(frame_rgb, text, (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)            
            
            # OpenCV로 텍스트 추가
            text = f"[pos: {next_state[0]:.2f}, theta_1: {next_state[2]*180.0/np.pi:.1f}, theta_2: {next_state[4]*180.0/np.pi:.1f}]"
            cv2.putText(frame_rgb, text, (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)                      

            # VideoWriter에 프레임 추가
            out.write(frame_rgb)

            step += 1            

        env.close()
        out.release()  # 비디오 파일 저장 종료
        # print(f"Video for {video_file_name} saved successfully with total reward displayed.")

if __name__ == "__main__":
    
    result_folder_path = 'results'
    if os.path.exists(f"{current_dir}/{result_folder_path}"):
        pass
    else:
        print("No results yet")  
        sys.exit()    
    
    video_folder_path = 'videos'
    performance_checker = PerformanceChecker(video_folder_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SerialDoublePendulumGym(render_mode="rgb_array")
    
    

        
            
    model_files = [
        file for file in os.listdir(f"{current_dir}/{result_folder_path}")
        if file.startswith("policy_net_") and file.endswith(".pth")
    ]    

    for model_file in model_files:
        model_path = os.path.join(f"{current_dir}/{result_folder_path}", model_file)
        
        model = performance_checker.load_model(model_path, env.get_observation_space().shape[0], env.get_action_space().n, device)
        video_file_name = f"performance_{model_file.split('.')[0]}"  # 모델 파일 이름을 영상 이름으로 사용
        performance_checker.visualize_performance(env, model, device, video_file_name)
        print(f"[SUCCESS] Video for {model_file} as {video_file_name}.mp4")        
