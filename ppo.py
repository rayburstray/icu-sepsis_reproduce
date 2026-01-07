import gymnasium as gym
import icu_sepsis
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# 1. 自定义 Wrapper：将 info 中的 state_vector 提取出来作为 Observation
class SepsisVectorWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 根据你的日志，state_vector 长度为 47
        self.vector_dim = 47
        # 定义新的观测空间为 47 维的连续向量
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.vector_dim,), 
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 提取向量并转为 float32
        vector_obs = info['state_vector'].astype(np.float32)
        return vector_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 提取向量并转为 float32
        vector_obs = info['state_vector'].astype(np.float32)
        return vector_obs, reward, terminated, truncated, info

def make_env():
    # 创建环境
    env = gym.make('Sepsis/ICU-Sepsis-v2')
    # 套上我们的 Wrapper
    env = SepsisVectorWrapper(env)
    return env

if __name__ == "__main__":
    # 检查 CUDA 是否可用
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 创建向量化环境 (SB3 要求)
    env = DummyVecEnv([make_env])

    # 3. 初始化 PPO 模型
    model = PPO(
        "MlpPolicy",  # 使用多层感知机策略
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        device=device  # 指定使用 cuda:0
    )

    # 4. 开始训练
    print("Start training...")
    total_timesteps = 100000  # 根据需要调整步数
    model.learn(total_timesteps=total_timesteps)
    
    # 保存模型
    model.save("ppo_icu_sepsis")
    print("Training finished and model saved.")

    # 5. 测试模型
    print("\nTesting model...")
    test_env = make_env()
    obs, _ = test_env.reset()
    
    total_reward = 0
    done = False
    
    while not done:
        # deterministic=True 表示测试时使用确定性策略（不采样）
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        
        # 打印部分步骤信息
        # print(f"Action: {action}, Reward: {reward}, Sofa Score: {info.get('sofa_score', 'N/A')}")
        
        if terminated or truncated:
            done = True

    print(f"Test Episode Reward: {total_reward}")
