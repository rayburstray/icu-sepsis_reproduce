import gymnasium as gym
import icu_sepsis
import numpy as np
import torch
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor  # <--- 引入 Monitor

# 设置日志目录
LOG_DIR = "./training_logs/"
os.makedirs(LOG_DIR, exist_ok=True)

# 1. 自定义 Wrapper
class SepsisVectorWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.vector_dim = 47
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.vector_dim,), 
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        vector_obs = info['state_vector'].astype(np.float32)
        return vector_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        vector_obs = info['state_vector'].astype(np.float32)
        return vector_obs, reward, terminated, truncated, info

def make_env():
    env = gym.make('Sepsis/ICU-Sepsis-v2')
    env = SepsisVectorWrapper(env)
    # 2. 添加 Monitor Wrapper，将数据记录到文件
    # allow_early_resets=True 允许在未完成时重置，这对某些环境很重要
    env = Monitor(env, filename=os.path.join(LOG_DIR, "log")) 
    return env

def plot_training_curve(log_dir):
    """
    读取 Monitor 生成的 CSV 文件并使用 Seaborn 绘图
    """
    # Monitor 生成的文件通常是 log.monitor.csv
    file_path = os.path.join(log_dir, "log.monitor.csv")
    
    if not os.path.exists(file_path):
        print("未找到日志文件，无法绘图。")
        return

    # 读取数据，跳过前两行（第一行是元数据）
    df = pd.read_csv(file_path, skiprows=1)
    
    # df 的列通常是: r (reward), l (length), t (time)
    # 我们计算累积步数 (Cumulative Timesteps)
    df['timesteps'] = df['l'].cumsum()
    
    # 为了曲线更平滑，计算滑动平均 (Rolling Mean)
    # window=50 表示计算最近 50 个 episode 的平均奖励
    window_size = 50
    df['reward_smooth'] = df['r'].rolling(window=window_size).mean()
    
    # 开始绘图
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid")
    
    # 绘制原始奖励（透明度高一点，作为背景）
    sns.lineplot(data=df, x='timesteps', y='r', alpha=0.3, color='lightblue', label='Raw Reward')
    
    # 绘制平滑后的奖励（深色，作为主线）
    sns.lineplot(data=df, x='timesteps', y='reward_smooth', color='blue', label=f'Smoothed Reward (MA-{window_size})')
    
    plt.title("PPO Training Curve: ICU Sepsis")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.tight_layout()
    
    # 显示图像
    plt.show()

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 创建环境
    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        device=device
    )

    print("Start training...")
    # 训练步数稍微调大一点，以便能画出好看的曲线
    total_timesteps = 100000 
    model.learn(total_timesteps=total_timesteps)
    
    model.save("ppo_icu_sepsis")
    print("Training finished.")

    # 3. 训练结束后调用绘图函数
    print("Plotting training curve...")
    plot_training_curve(LOG_DIR)
