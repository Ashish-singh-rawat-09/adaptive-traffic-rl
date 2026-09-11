import os
import sys
import matplotlib
matplotlib.use('Agg')  # Fix for Tcl/Tk issue on Windows
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.envs.traffic_env import SumoTrafficEnv
from src.agents.baselines import FixedTimeAgent


def run_simulation(agent, env, total_steps=80):
    obs, _ = env.reset()
    delays = []
    queues = []

    for _ in range(total_steps):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        total_queue = sum(obs)
        delays.append(-reward)
        queues.append(total_queue)

        if terminated or truncated:
            break

    env.close()
    return delays, queues


def main():
    net_file = "data/networks/intersection.net.xml"
    route_file = "data/routes/normal_hours.rou.xml"
    model_path = "models/final_policy.zip"

    print("[BENCHMARK] Running Fixed-Time Controller baseline...")
    env_fixed = SumoTrafficEnv(net_file, route_file, use_gui=False, sim_max_time=1000)
    fixed_agent = FixedTimeAgent(switch_interval=3)
    fixed_delays, fixed_queues = run_simulation(fixed_agent, env_fixed)

    print("[BENCHMARK] Running Trained DQN Agent...")
    env_rl = SumoTrafficEnv(net_file, route_file, use_gui=False, sim_max_time=1000)
    rl_agent = DQN.load(model_path)
    rl_delays, rl_queues = run_simulation(rl_agent, env_rl)

    avg_delay_fixed = sum(fixed_delays) / len(fixed_delays)
    avg_delay_rl = sum(rl_delays) / len(rl_delays)
    improvement = ((avg_delay_fixed - avg_delay_rl) / avg_delay_fixed) * 100

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 50)
    print(f"Fixed-Time Signal Avg Penalty : {avg_delay_fixed:.2f}")
    print(f"RL Adaptive Signal Avg Penalty: {avg_delay_rl:.2f}")
    print(f"Traffic Efficiency Gain       : {improvement:.2f}% improvement")
    print("=" * 50)

    plt.figure(figsize=(12, 5))

    # Delay Comparison
    plt.subplot(1, 2, 1)
    plt.plot(fixed_delays, label="Fixed-Time (Traditional)", color="red", linestyle="--")
    plt.plot(rl_delays, label="Adaptive RL (DQN)", color="green")
    plt.title("Cumulative Traffic Delay Over Time")
    plt.xlabel("Control Steps (10s intervals)")
    plt.ylabel("Delay / Waiting Penalty")
    plt.legend()
    plt.grid(True)

    # Queue Length Comparison
    plt.subplot(1, 2, 2)
    plt.plot(fixed_queues, label="Fixed-Time Queue", color="crimson", linestyle="--")
    plt.plot(rl_queues, label="RL Adaptive Queue", color="teal")
    plt.title("Total Halting Vehicles (Queue Length)")
    plt.xlabel("Control Steps (10s intervals)")
    plt.ylabel("Vehicles Waiting")
    plt.legend()
    plt.grid(True)

    os.makedirs("notebooks", exist_ok=True)
    graph_path = "notebooks/rl_vs_fixed_benchmark.png"
    plt.tight_layout()
    plt.savefig(graph_path)
    print(f"[SAVED] Comparison graph saved at: {graph_path}")
    plt.show()


if __name__ == "__main__":
    main()