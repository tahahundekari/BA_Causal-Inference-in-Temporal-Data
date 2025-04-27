import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os

def generate_synthetic_data_3vars(n_timesteps=300, n_vars=3, noise_std=0.1, intervention_rate=0.1, random_seed=42):
    np.random.seed(random_seed)

    X = np.zeros((n_timesteps, n_vars))
    interventions = np.zeros((n_timesteps, n_vars), dtype=int)

    # Define ground-truth temporal dynamics (only t-1 -> t)
    A_temporal = np.array([
        [0.7, 0.0, 0.0],  # X1(t-1) -> X1(t)
        [0.5, 0.6, 0.0],  # X1(t-1) and X2(t-1) -> X2(t)
        [0.0, 0.4, 0.7]   # X2(t-1) and X3(t-1) -> X3(t)
    ])

    # Random interventions
    for t in range(1, n_timesteps):
        if np.random.rand() < intervention_rate:
            var_to_intervene = np.random.choice(n_vars)
            interventions[t, var_to_intervene] = 1

    # Simulate data
    for t in range(1, n_timesteps):
        X[t] = A_temporal @ X[t-1]  # Temporal dynamics only

        for var in range(n_vars):
            if interventions[t, var] == 1:
                X[t, var] = np.random.randn()  # Random intervention

        X[t] += np.random.normal(0, noise_std, size=n_vars)  # Add observation noise

    return X, interventions

def generate_synthetic_data_5vars(n_timesteps=300, n_vars=5, noise_std=0.1, intervention_rate=0.1, random_seed=42):
    np.random.seed(random_seed)

    X = np.zeros((n_timesteps, n_vars))
    interventions = np.zeros((n_timesteps, n_vars), dtype=int)

    # Define ground-truth temporal dynamics (only t-1 -> t)
    A_temporal = np.array([
        [0.7, 0.0, 0.0, 0.0, 0.0],  # X1(t-1) -> X1(t)
        [0.4, 0.6, 0.0, 0.0, 0.0],  # X1(t-1) and X2(t-1) -> X2(t)
        [0.0, 0.5, 0.5, 0.0, 0.0],  # X2(t-1) and X3(t-1) -> X3(t)
        [0.0, 0.0, 0.6, 0.4, 0.0],  # X3(t-1) and X4(t-1) -> X4(t)
        [0.2, 0.0, 0.0, 0.5, 0.6]   # X1(t-1), X4(t-1), and X5(t-1) -> X5(t)
    ])

    for t in range(1, n_timesteps):
        if np.random.rand() < intervention_rate:
            var_to_intervene = np.random.choice(n_vars)
            interventions[t, var_to_intervene] = 1

    for t in range(1, n_timesteps):
        X[t] = A_temporal @ X[t-1]

        for var in range(n_vars):
            if interventions[t, var] == 1:
                X[t, var] = np.random.randn()

        X[t] += np.random.normal(0, noise_std, size=n_vars)

    return X, interventions

def save_data(X, interventions, output_prefix):
    df_X = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(X.shape[1])])
    df_interv = pd.DataFrame(interventions, columns=[f'Intervene_X{i+1}' for i in range(X.shape[1])])

    df_X.to_csv(os.path.join("data", "raw", "linear", f"{output_prefix}_observations.csv"), index=False)
    df_interv.to_csv(os.path.join("data", "raw", "linear", f"{output_prefix}_interventions.csv"), index=False)
    print(f"Data saved to {output_prefix}_observations.csv and {output_prefix}_interventions.csv")

def plot_causal_graph_3vars():
    G = nx.DiGraph()
    G.add_edge('X1(t-1)', 'X1(t)')
    G.add_edge('X1(t-1)', 'X2(t)')
    G.add_edge('X2(t-1)', 'X2(t)')
    G.add_edge('X2(t-1)', 'X3(t)')
    G.add_edge('X3(t-1)', 'X3(t)')

    pos = {
        'X1(t-1)': (-2, 1), 'X2(t-1)': (-2, 0), 'X3(t-1)': (-2, -1),
        'X1(t)': (2, 1), 'X2(t)': (2, 0), 'X3(t)': (2, -1)
    }
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=2000, font_size=10)
    plt.title("Ground-Truth Causal Graph (3 Variables, First-Order Markov, Linear)")
    plt.savefig(os.path.join("data", "raw", "linear", "causal_graph_3vars.png"), dpi=300, bbox_inches='tight')

def plot_causal_graph_5vars():
    G = nx.DiGraph()
    G.add_edge('X1(t-1)', 'X1(t)')
    G.add_edge('X1(t-1)', 'X2(t)')
    G.add_edge('X2(t-1)', 'X2(t)')
    G.add_edge('X2(t-1)', 'X3(t)')
    G.add_edge('X3(t-1)', 'X3(t)')
    G.add_edge('X3(t-1)', 'X4(t)')
    G.add_edge('X4(t-1)', 'X4(t)')
    G.add_edge('X4(t-1)', 'X5(t)')
    G.add_edge('X5(t-1)', 'X5(t)')
    G.add_edge('X1(t-1)', 'X5(t)')

    pos = {
        'X1(t-1)': (-2, 2), 'X2(t-1)': (-2, 1), 'X3(t-1)': (-2, 0), 'X4(t-1)': (-2, -1), 'X5(t-1)': (-2, -2),
        'X1(t)': (2, 2), 'X2(t)': (2, 1), 'X3(t)': (2, 0), 'X4(t)': (2, -1), 'X5(t)': (2, -2)
    }
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=2000, font_size=10)
    plt.title("Ground-Truth Causal Graph (5 Variables, First-Order Markov, Linear)")
    plt.savefig(os.path.join("data", "raw", "linear", "causal_graph_5vars.png"), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    X, interventions = generate_synthetic_data_3vars()
    save_data(X, interventions, "synthetic_data_3vars_markov")

    X, interventions = generate_synthetic_data_5vars()
    save_data(X, interventions, "synthetic_data_5vars_markov")

    plot_causal_graph_3vars()
    plot_causal_graph_5vars()
