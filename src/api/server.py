import os
import sys
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(
    title="Adaptive Traffic Signal RL API",
    description="Real-time traffic phase recommendation engine using ONNX RL policy",
    version="1.0.0"
)

ONNX_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../models/traffic_dqn.onnx")
)

# Load ONNX session once at startup
if os.path.exists(ONNX_MODEL_PATH):
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name
else:
    session = None
    input_name = None

class TrafficStateRequest(BaseModel):
    lane_queues: List[float] = Field(
        ...,
        min_length=8,
        max_length=8,
        description="Vehicle queue lengths for 8 incoming lanes [N_L1, N_L2, S_L1, S_L2, E_L1, E_L2, W_L1, W_L2]"
    )

class PhaseResponse(BaseModel):
    recommended_phase: int
    phase_name: str
    action_q_values: List[float]

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": session is not None,
        "model_path": ONNX_MODEL_PATH
    }

@app.post("/predict_phase", response_model=PhaseResponse)
def predict_phase(payload: TrafficStateRequest):
    if session is None:
        raise HTTPException(status_code=500, detail="ONNX model not loaded on server.")

    # Prepare input tensor: shape (1, 8)
    input_data = np.array([payload.lane_queues], dtype=np.float32)

    # Run inference
    q_values = session.run(None, {input_name: input_data})[0][0]
    best_action = int(np.argmax(q_values))

    phase_label = "North-South Green" if best_action == 0 else "East-West Green"

    return PhaseResponse(
        recommended_phase=best_action,
        phase_name=phase_label,
        action_q_values=[float(q) for q in q_values]
    )