"""
Run Kalman Filter on Preprocessed Pong
"""
import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import os
import argparse
from datetime import datetime


def load_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    
    if "images" in data.files:
        images = data["images"]
        if images.ndim == 4:
            images = images.reshape(images.shape[0], -1)
        elif images.ndim > 2:
            images = images.reshape(images.shape[0], -1)
            if images.shape[1] != 32:
                print(f"Warning: Reshaped images to shape {images.shape}")
        
        print(f"Loaded images with shape: {images.shape}")
        return images
    else:
        raise ValueError(f"Could not find 'images' in {npz_path}. Available keys: {data.files}")


def compute_metrics(predictions, targets, keys=None):
    n_samples_pred = predictions.shape[0]
    n_samples_targets = targets.shape[0]
    
    if n_samples_pred != n_samples_targets:
        min_samples = min(n_samples_pred, n_samples_targets)
        predictions = predictions[:min_samples]
        targets = targets[:min_samples]
        print(f"Trimmed both arrays to {min_samples} samples.")
    
    dim = predictions.shape[1]
    
    keys = [f"dim_{i}" for i in range(dim)]
    
    spearman_corr = np.zeros((dim, dim))
    r2_mat = np.zeros((dim, dim))
    mse_mat = np.zeros((dim, dim))
    
    for i in range(dim):
        for j in range(dim):
            spearman_corr[i, j] = spearmanr(predictions[:, i], targets[:, j])[0]
            r2_mat[i, j] = r2_score(targets[:, j], predictions[:, i])
            mse_mat[i, j] = mean_squared_error(targets[:, j], predictions[:, i])
    
    return spearman_corr, r2_mat, mse_mat, keys


def apply_kalman_filter(train_data, test_data, n_iter=10, transition_cov_scale=0.01, observation_cov_scale=0.1):
    print(f"Applying Kalman filter with {n_iter} iterations...")
    
    state_dim = train_data.shape[1]  # 32
    
    print(f"Data shape: {train_data.shape}, State dimension: {state_dim}")
    
    kf = KalmanFilter(
        transition_matrices=np.eye(state_dim),
        observation_matrices=np.eye(state_dim),
        transition_covariance=transition_cov_scale * np.eye(state_dim),
        observation_covariance=observation_cov_scale * np.eye(state_dim),
        initial_state_mean=train_data[0],
        initial_state_covariance=np.eye(state_dim)
    )
    
    print("Training Kalman filter using EM algorithm...")
    kf = kf.em(train_data, n_iter=n_iter)
    print("EM training complete.")
    
    print("Filtering test data...")
    filtered_state_means, filtered_state_covs = kf.filter(test_data)
    
    print("Generating one-step-ahead predictions...")
    predicted_next_states = np.zeros((test_data.shape[0]-1, state_dim))
    
    for t in range(test_data.shape[0]-1):
        next_state_mean, _ = kf.filter_update(
            filtered_state_mean=filtered_state_means[t],
            filtered_state_covariance=filtered_state_covs[t]
        )
        predicted_next_states[t] = next_state_mean
    
    filter_mse = mean_squared_error(test_data, filtered_state_means)
    prediction_mse = mean_squared_error(test_data[1:], predicted_next_states)

    error_images = np.abs(test_data[1:] - predicted_next_states)

    mse_per_image = np.mean(error_images, axis=1)
    
    print(f"Current frame reconstruction MSE: {filter_mse:.6f}")
    print(f"Next frame prediction MSE: {prediction_mse:.6f}")
    
    return {
        'filtered_state_means': filtered_state_means,
        'filtered_state_covs': filtered_state_covs,
        'predicted_next_states': predicted_next_states,
        'filter_mse': filter_mse,
        'error_images': error_images,
        'mse_per_image': mse_per_image,
        'prediction_mse': prediction_mse,
        'kalman_filter': kf
    }


def plot_kalman_results(results, output_dir):
    """Plot the results of Kalman filtering with focus on predictions and errors"""
    os.makedirs(output_dir, exist_ok=True)
    
    test_data = results['test_data']
    filtered_means = results['filtered_state_means']
    filtered_covs = results['filtered_state_covs']
    predicted_next = results['predicted_next_states']
    
    n_dims = test_data.shape[1]  # 32
    
    keys = [f"Column {i}" for i in range(n_dims)]
    
    plt.figure(figsize=(20, 15))
    for i in range(n_dims):
        plt.subplot(8, 4, i+1)
        
        plt.plot(test_data[:100, i], 'b-', label='Original', alpha=0.7)
        plt.plot(filtered_means[:100, i], 'r-', label='Filtered')
        
        plt.title(f'Column {i}', fontsize=10)
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_filtered_states.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(20, 15))
    for i in range(n_dims):
        plt.subplot(8, 4, i+1)
        
        plt.plot(test_data[1:100, i], 'b-', label='Actual', alpha=0.7)
        plt.plot(predicted_next[:99, i], 'g-', label='Predicted')
        
        plt.title(f'Column {i} Prediction', fontsize=10)
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_predictions.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(20, 15))
    for i in range(n_dims):
        plt.subplot(8, 4, i+1)
        
        error = test_data[1:100, i] - predicted_next[:99, i]
        
        plt.plot(error, 'r-')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.title(f'Column {i} Error', fontsize=10)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_prediction_errors.png"), dpi=300)
    plt.close()
    
    mse_per_dim = np.mean((test_data[1:] - predicted_next) ** 2, axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(n_dims), mse_per_dim)
    plt.xlabel('Column Index')
    plt.ylabel('Mean Squared Error')
    plt.title('Prediction MSE by Column')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse_by_column.png"), dpi=300)
    plt.close()
    
    transition_matrix = results['kalman_filter'].transition_matrices
    plt.figure(figsize=(12, 10))
    plt.imshow(transition_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Value')
    plt.title('Learned Transition Matrix')
    
    plt.xlabel('From Column')
    plt.ylabel('To Column')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transition_matrix.png"), dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kalman Filter on Pong column data")
    parser.add_argument('--train_path', type=str, required=True, help='Path to training data .npz file')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test data .npz file')
    parser.add_argument('--output_dir', type=str, default='checkpoints/KalmanFilter', help='Directory to save output')
    parser.add_argument('--experiment_name', type=str, default='Pong_unified_kalman', help='Name for this experiment')
    parser.add_argument('--n_iter', type=int, default=10, help='Number of EM iterations for Kalman filter')
    parser.add_argument('--transition_cov', type=float, default=0.01, help='Transition covariance scale')
    parser.add_argument('--observation_cov', type=float, default=0.1, help='Observation covariance scale')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (None uses all data)')
    
    args = parser.parse_args()
    
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    versions = [d for d in os.listdir(experiment_dir) if d.startswith('version_')]
    if versions:
        version_nums = [int(v.split('_')[1]) for v in versions]
        version_num = max(version_nums) + 1
    else:
        version_num = 0
    
    version_dir = os.path.join(experiment_dir, f"version_{version_num}")
    checkpoint_dir = os.path.join(version_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Results will be saved to: {version_dir}")
    
    print(f"Loading training data from {args.train_path}")
    train_data = load_data(args.train_path)
    
    print(f"Loading test data from {args.test_path}")
    test_data = load_data(args.test_path)
    
    if train_data.shape[1] != 32:
        print(f"Warning: Expected training data with 32 dimensions, got {train_data.shape[1]}.")
    
    if test_data.shape[1] != 32:
        print(f"Warning: Expected test data with 32 dimensions, got {test_data.shape[1]}.")
    
    if args.batch_size is not None:
        if args.batch_size < len(train_data):
            print(f"Using {args.batch_size} samples from training data (out of {len(train_data)})")
            train_data = train_data[:args.batch_size]
    
    start_time = datetime.now()
    
    kalman_results = apply_kalman_filter(
        train_data, 
        test_data,
        n_iter=args.n_iter,
        transition_cov_scale=args.transition_cov,
        observation_cov_scale=args.observation_cov
    )

    end_time = datetime.now()
    
    kalman_results['test_data'] = test_data
    
    vis_dir = os.path.join(version_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    plot_kalman_results(kalman_results, vis_dir)
    
    hparams = {
        "model_type": "KalmanFilter",
        "train_path": args.train_path,
        "test_path": args.test_path,
        "n_iter": args.n_iter,
        "transition_cov_scale": args.transition_cov,
        "observation_cov_scale": args.observation_cov,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "state_dimensions": train_data.shape[1],
        "filter_mse": float(kalman_results['filter_mse']),
        "prediction_mse": float(kalman_results['prediction_mse']),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(version_dir, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2)
    
    comparison_results = {
        "filter_mse": float(kalman_results['filter_mse']),
        "prediction_mse": float(kalman_results['prediction_mse']),
        "num_features": train_data.shape[1],
        "num_samples": len(test_data),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(version_dir, "comparison_results.json"), "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    np.savez_compressed(
        os.path.join(checkpoint_dir, "kalman_results.npz"),
        train_data=train_data[:100],
        test_data=test_data[:100],
        filtered_means=kalman_results['filtered_state_means'],
        filtered_covs=kalman_results['filtered_state_covs'],
        predicted_next=kalman_results['predicted_next_states'],
        error_images=kalman_results['error_images'],
        mse_per_image=kalman_results['mse_per_image'],
        filter_mse=kalman_results['filter_mse'],
        prediction_mse=kalman_results['prediction_mse'],
        transition_matrix=kalman_results['kalman_filter'].transition_matrices
    )
    
    with open(os.path.join(version_dir, "summary.txt"), "w") as f:
        f.write(f"Kalman Filter Analysis\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Training data: {args.train_path}\n")
        f.write(f"Test data: {args.test_path}\n")
        f.write(f"State dimensions: {train_data.shape[1]}\n")
        f.write(f"Training samples: {len(train_data)}\n")
        f.write(f"Test samples: {len(test_data)}\n")
        f.write(f"Kalman EM iterations: {args.n_iter}\n\n")
        f.write(f"Current frame reconstruction MSE: {kalman_results['filter_mse']:.6f}\n")
        f.write(f"Next frame prediction MSE: {kalman_results['prediction_mse']:.6f}\n")
        f.write(f"Total time taken: {end_time - start_time}\n")
    
    print(f"\nAll results saved to {version_dir}")
    print(f"Filter MSE: {kalman_results['filter_mse']:.6f}")
    print(f"Prediction MSE: {kalman_results['prediction_mse']:.6f}")
