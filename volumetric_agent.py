# volumetric_agent.py
from mcp.server.fastmcp import FastMCP
import lightgbm as lgb
import numpy as np
import os

mcp = FastMCP("Volumetric_Expert_Agent")

agent_state = {
    "trust_score": 1.0,  
    "model": None 
}

# UPDATE: Point to the new text model file!
MODEL_PATH = "lightgbm_global_model.txt"
if os.path.exists(MODEL_PATH):
    agent_state["model"] = lgb.Booster(model_file=MODEL_PATH)

@mcp.tool()
def evaluate_threat(features: list[float]) -> float:
    if agent_state["model"] is None: return 0.5
    
    # THE ALIGNMENT FIX: Drop src_port and dst_port (the first 2 features)
    if len(features) == 79:
        features = features[2:]
        
    data = np.array([features])
    prob = agent_state["model"].predict(data)[0]
    return float(prob)

@mcp.tool()
def get_trust_score() -> float:
    return agent_state["trust_score"]

@mcp.tool()
def update_trust(reward: int) -> str:
    alpha = 0.1 
    old_trust = agent_state["trust_score"]
    agent_state["trust_score"] = old_trust + alpha * (reward - old_trust)
    return f"Trust updated to {agent_state['trust_score']:.4f}"

if __name__ == "__main__":
    mcp.run()
