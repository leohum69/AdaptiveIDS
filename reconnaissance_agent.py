# recon_agent.py
# recon_agent.py
from mcp.server.fastmcp import FastMCP
import joblib
import numpy as np

mcp = FastMCP("Recon_Expert_Agent")

# Load the Brain and the Scaler!
agent_state = {
    "trust_score": 1.0,
    "model": joblib.load("rf_model.pkl"),
    "scaler": joblib.load("scaler_agents.pkl")
}

@mcp.tool()
def evaluate_threat(features: list[float]) -> float:
    # 1. Convert live data to numpy array
    data = np.array([features])
    
    # 2. Scale the live data using the saved scaler
    scaled_data = agent_state["scaler"].transform(data)
    
    # 3. Predict the probability of an attack
    prob = agent_state["model"].predict_proba(scaled_data)[0][1]
    return float(prob)

@mcp.tool()
def get_trust_score() -> float:
    return agent_state["trust_score"]

@mcp.tool()
def update_trust(reward: int) -> str:
    """Updates trust based on Active Learning ground truth feedback."""
    alpha = 0.1 
    old_trust = agent_state["trust_score"]
    
    # RL Trust Update Formula
    new_trust = old_trust + alpha * (reward - old_trust)
    agent_state["trust_score"] = new_trust
    
    return f"Recon Agent trust adjusted to {new_trust:.4f}"

if __name__ == "__main__":
    mcp.run()
