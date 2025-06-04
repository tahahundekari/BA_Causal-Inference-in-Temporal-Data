import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from sklearn.metrics import mean_squared_error
import os

def run_kalman_filter(obs_path, output_prefix):
    df = pd.read_csv(obs_path)
    time = df["time"].values
    data = df.drop(columns=["time"]).values

    n_timesteps, n_vars = data.shape
    split_idx = n_timesteps // 2

    train_data = data[:split_idx]
    test_data = data[split_idx:]
    test_time = time[split_idx:]

    kf = KalmanFilter(
        transition_matrices=np.eye(n_vars),
        observation_matrices=np.eye(n_vars),
        transition_covariance=0.01 * np.eye(n_vars),
        observation_covariance=0.1 * np.eye(n_vars),
        initial_state_mean=np.zeros(n_vars),
        initial_state_covariance=np.eye(n_vars)
    )
    kf = kf.em(train_data, n_iter=10)

    filtered_state_means, _ = kf.filter(data)
    smoothed_state_means, _ = kf.smooth(data)

    X_true = test_data
    X_filt = filtered_state_means[split_idx:]

    rmse_filt = np.sqrt(mean_squared_error(X_true, X_filt))

    print(f"{output_prefix} Filtered RMSE (Test): {rmse_filt:.4f}")

    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(14, 10))
    for i in range(n_vars):
        plt.subplot(n_vars, 1, i + 1)
        plt.plot(test_time, X_true[:, i], label='Observed (Test)')
        plt.plot(test_time, X_filt[:, i], label='Filtered')
        plt.title(f'Variable X{i+1}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"results/{output_prefix.lower()}_kalman_test_plot.png", dpi=300)

if __name__ == "__main__":
    run_kalman_filter(
        "data/raw/structured_linear/structured_latent_observations.csv",
        output_prefix="Kalman_3vars"
    )

    # run_kalman_filter(
    #     "data/raw/linear/synthetic_data_5vars_markov_observations.csv",
    #     output_prefix="Kalman_5vars"
    # )
