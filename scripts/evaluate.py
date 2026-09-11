import os
import sys
import time
from stable_baselines3 import DQN

# Project root path add karna
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.envs.traffic_env import SumoTrafficEnv


def main():
    net_file = "data/networks/intersection.net.xml"
    route_file = "data/routes/normal_hours.rou.xml"
    model_path = "models/final_policy.zip"

    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model nahi mila: {model_path}")
        print("Pehle training complete hone do!")
        return

    print("[INFO] Launching SUMO-GUI with trained RL agent...")
    # use_gui=True taaki screen par live visually dikhe
    env = SumoTrafficEnv(
        net_file=net_file,
        route_file=route_file,
        use_gui=True,
        sim_max_time=1000,
    )

    print(f"[INFO] Loading trained model from {model_path}...")
    model = DQN.load(model_path)

    obs, _ = env.reset()
    total_reward = 0.0
    step_count = 0

    print("[INFO] Simulation started. Press Ctrl+C in terminal to stop early.")
    try:
        while True:
            # Deterministic=True se model best learned action choose karega
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if terminated or truncated:
                print(
                    f"\n[EVALUATION COMPLETE] Total Steps: {step_count}, Total Cumulative Reward: {total_reward:.2f}"
                )
                break
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()