import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Smart Traffic RL Controller", layout="wide")

st.title("🚦 Adaptive Traffic Signal Control - Live RL Dashboard")
st.markdown("Dynamic Traffic Light Phase Prediction using Deep Q-Networks (ONNX Inference)")

st.sidebar.header("🕹️ Incoming Lane Traffic (Vehicle Counts)")
n1 = st.sidebar.slider("North Lane 1", 0, 30, 8)
n2 = st.sidebar.slider("North Lane 2", 0, 30, 4)
s1 = st.sidebar.slider("South Lane 1", 0, 30, 2)
s2 = st.sidebar.slider("South Lane 2", 0, 30, 1)
e1 = st.sidebar.slider("East Lane 1", 0, 30, 15)
e2 = st.sidebar.slider("East Lane 2", 0, 30, 10)
w1 = st.sidebar.slider("West Lane 1", 0, 30, 6)
w2 = st.sidebar.slider("West Lane 2", 0, 30, 5)

lane_queues = [float(x) for x in [n1, n2, s1, s2, e1, e2, w1, w2]]

api_url = "http://127.0.0.1:8000/predict_phase"
try:
    response = requests.post(api_url, json={"lane_queues": lane_queues})
    if response.status_code == 200:
        data = response.json()
        rec_phase = data["recommended_phase"]
        phase_name = data["phase_name"]
        q_vals = data["action_q_values"]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Signal Status")
            if rec_phase == 0:
                st.success(f"🟢 **Active: {phase_name}**")
                st.error("🔴 East-West: RED")
            else:
                st.success(f"🟢 **Active: {phase_name}**")
                st.error("🔴 North-South: RED")

            fig_q = go.Figure(data=[
                go.Bar(
                    x=["North-South Green", "East-West Green"],
                    y=q_vals,
                    marker_color=["#1f77b4", "#ff7f0e"]
                )
            ])
            fig_q.update_layout(title="Action Q-Values (Higher is better)", yaxis_title="Estimated Return")
            st.plotly_chart(fig_q, use_container_width=True)

        with col2:
            st.subheader("Approach Traffic Distribution")
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=["North", "South", "East", "West"],
                    y=[n1 + n2, s1 + s2, e1 + e2, w1 + w2],
                    marker_color=["#4C78A8", "#72B7B2", "#E45756", "#F58518"]
                )
            ])
            fig_bar.update_layout(title="Total Halting Vehicles per Approach", yaxis_title="Vehicles")
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.error(f"API Error: {response.status_code}")
except Exception as e:
    st.warning("Ensure FastAPI server (src.api.server:app) is running on port 8000!")
