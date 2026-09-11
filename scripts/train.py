import os
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# Project root path add karna
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.envs.traffic_env import SumoTrafficEnv

def main():
    net_file = "data/networks/intersection.net.xml"
    route_file = "data/routes/normal_hours.rou.xml"

    print("[INFO] Initializing SumoTrafficEnv for training...")
    # Training headless (use_gui=False) mode me tez hoti hai
    env = SumoTrafficEnv(net_file=net_file, route_file=route_file, use_gui=False, sim_max_time=1000)

    # Checkpoints save karne ke liye callback
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./models/checkpoints/",
        name_prefix="dqn_traffic_model"
    )

    print("[INFO] Initializing DQN Agent...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=500,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./notebooks/logs/"
    )

    # Total timesteps set karo (Initial test run ke liye 10,000 steps)
    total_timesteps = 10000
    print(f"[INFO] Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Final policy save karna
    os.makedirs("models", exist_ok=True)
    save_path = "models/final_policy"
    model.save(save_path)
    print(f"[SUCCESS] Model successfully saved at {save_path}.zip")

    env.close()

if __name__ == "__main__":
    main()