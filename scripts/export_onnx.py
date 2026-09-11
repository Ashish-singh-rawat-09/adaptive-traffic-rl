import os
import sys
import torch
from stable_baselines3 import DQN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    model_path = "models/final_policy.zip"
    onnx_output_path = "models/traffic_dqn.onnx"

    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model file nahi mili: {model_path}")
        return

    print(f"[INFO] Loading SB3 DQN model from {model_path}...")
    model = DQN.load(model_path)

    # Policy network extract karna
    policy_net = model.policy.q_net

    # Dummy input representing the 8 incoming lane observations
    dummy_input = torch.randn(1, 8, dtype=torch.float32)

    print(f"[INFO] Exporting to ONNX: {onnx_output_path}...")
    torch.onnx.export(
        policy_net,
        dummy_input,
        onnx_output_path,
        opset_version=14,
        input_names=["lane_queues"],
        output_names=["action_q_values"],
        dynamic_axes={
            "lane_queues": {0: "batch_size"},
            "action_q_values": {0: "batch_size"}
        }
    )

    print(f"[SUCCESS] ONNX model exported cleanly to: {onnx_output_path}")

if __name__ == "__main__":
    main()