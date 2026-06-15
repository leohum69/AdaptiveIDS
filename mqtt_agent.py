# mqtt_agent.py
from mcp.server.fastmcp import FastMCP
import joblib
import numpy as np
import os

mcp = FastMCP("MQTT_Expert_Agent")

agent_state = {
    "trust_score": 1.0,
    "model": None
}

MODEL_PATH = "xgboost_mqtt_intrusion_model.pkl"
if os.path.exists(MODEL_PATH):
    agent_state["model"] = joblib.load(MODEL_PATH)
else:
    print(f"Warning: {MODEL_PATH} not found.")

@mcp.tool()
def evaluate_threat(features: list[float]) -> float:
    if agent_state["model"] is None: return 0.5
    
    # THE ALIGNMENT FIX: Drop src_port and dst_port (the first 2 features)
    if len(features) == 79:
        features = features[2:]
        
    data = np.array([features])
    prob = agent_state["model"].predict_proba(data)[0][1]
    return float(prob)

@mcp.tool()
def get_trust_score() -> float:
    return agent_state["trust_score"]

@mcp.tool()
def update_trust(reward: int) -> str:
    alpha = 0.1
    old_trust = agent_state["trust_score"]
    new_trust = old_trust + alpha * (reward - old_trust)
    agent_state["trust_score"] = new_trust
    return f"Trust updated to {new_trust:.4f}"

if __name__ == "__main__":
    mcp.run()
