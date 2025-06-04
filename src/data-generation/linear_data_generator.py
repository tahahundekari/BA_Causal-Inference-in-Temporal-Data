import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os

def generate_structured_latent_data(n_timesteps=300, n_obs=3, n_latents=3, noise_std=0.1, seed=42):
    np.random.seed(seed)

    Z = np.zeros((n_timesteps, n_latents))  # Causal variables C^t

    # Temporal structure: Z_t = A * Z_{t-1} + noise
    A_temporal = np.eye(n_latents) * 0.6
    A_temporal[1, 0] = 0.3  # C1^t -> C2^t+1
    A_temporal[2, 0] = 0.2  # C1^t -> C3^t+1
    A_temporal[2, 1] = 0.2  # C2^t -> C3^t+1

    # Observation matrix: X = C * Z + noise
    C_obs = np.random.randn(n_obs, n_latents)

    for t in range(1, n_timesteps):
        Z[t] = A_temporal @ Z[t-1]  # temporal causality
        Z[t] += np.random.normal(0, noise_std, size=n_latents)

    # Observations with measurement noise
    X = (C_obs @ Z.T).T + np.random.normal(0, noise_std, size=(n_timesteps, n_obs))

    return X, Z, C_obs, A_temporal

def save_data_npz(X, Z):
    os.makedirs("data/raw/structured_linear", exist_ok=True)
    
    # Create time array
    time = np.arange(len(X))
    
    # Save to NPZ format (will contain multiple arrays in one file)
    np.savez(
        "data/raw/structured_linear/train.npz",
        observations=X,
        latents=Z,
        time=time
    )
    
    print(f"Saved data as train.npz in data/raw/structured_linear/")

def plot_causal_graph():
    G = nx.DiGraph()
    # Temporal: Z(t-1) -> Z(t)
    G.add_edge("Z1(t-1)", "Z1(t)")
    G.add_edge("Z1(t-1)", "Z2(t)")
    G.add_edge("Z2(t-1)", "Z2(t)")
    G.add_edge("Z2(t-1)", "Z3(t)")
    G.add_edge("Z3(t-1)", "Z3(t)")

    pos = {
        "Z1(t-1)": (0, 6), "Z2(t-1)": (0, 4), "Z3(t-1)": (0, 2),
        "Z1(t)": (1, 6), "Z2(t)": (1, 4), "Z3(t)": (1, 2)
    }

    plt.figure(figsize=(6, 8))
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=2500, font_size=12)
    plt.title("Ground-Truth Latent Causal Graph")
    plt.savefig("data/raw/structured_linear/latent_temporal_graph.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    X, Z, C_obs, A = generate_structured_latent_data()
    save_data_npz(X, Z)
    plot_causal_graph()
