"""
Running the Kalman Filter on the Pong ground-truth variables.
For fun.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import os
from datetime import datetime


def load_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    latents = data['latents']
    keys = data['keys']
    return latents, keys


def compute_metrics(latents, targets, keys):
    n_samples_latents = latents.shape[0]
    n_samples_targets = targets.shape[0]
    
    if n_samples_latents != n_samples_targets:
        print(f"WARNING: Shape mismatch between latents ({n_samples_latents} samples) "
              f"and targets ({n_samples_targets} samples).")
        
        min_samples = min(n_samples_latents, n_samples_targets)
        latents = latents[:min_samples]
        targets = targets[:min_samples]
        print(f"Trimmed both arrays to {min_samples} samples.")
    
    spearman_corr = np.zeros((len(keys), len(keys)))
    r2_mat = np.zeros((len(keys), len(keys)))
    
    for i, _ in enumerate(keys):
        for j, _ in enumerate(keys):
            y_pred = latents[:, i]
            y_true = targets[:, j]
            
            try:
                corr = spearmanr(y_pred, y_true).correlation
                spearman_corr[i, j] = 0.0 if np.isnan(corr) else corr
            except:
                spearman_corr[i, j] = 0.0
                
            try:
                r2_mat[i, j] = r2_score(y_true, y_pred)
            except:
                r2_mat[i, j] = 0.0
    
    return spearman_corr, r2_mat

def print_matrix(matrix, keys, title):
    df = pd.DataFrame(matrix, index=keys, columns=keys)
    print(f"\n{title}:\n")
    print(df.round(3))
    print()

def apply_kalman_filter(train_latents, test_latents, n_iter=10, transition_cov_scale=0.01, observation_cov_scale=0.1):
    print(f"Applying Kalman filter with {n_iter} iterations...")
    
    n_vars = train_latents.shape[1]
    
    kf = KalmanFilter(
        transition_matrices=np.eye(n_vars),
        observation_matrices=np.eye(n_vars),
        transition_covariance=transition_cov_scale * np.eye(n_vars),
        observation_covariance=observation_cov_scale * np.eye(n_vars),
        initial_state_mean=np.zeros(n_vars),
        initial_state_covariance=np.eye(n_vars)
    )
    
    print("Fitting Kalman filter on training data...")
    kf = kf.em(train_latents, n_iter=n_iter)
    
    print("Filtering training data...")
    train_filtered_means, train_filtered_covs = kf.filter(train_latents)
    
    print("Applying filter to test data...")
    test_filtered_means, test_filtered_covs = kf.filter(test_latents)
    
    train_rmse = np.sqrt(mean_squared_error(train_latents, train_filtered_means))
    test_rmse = np.sqrt(mean_squared_error(test_latents, test_filtered_means))
    
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    return {
        'train_latents': train_latents,
        'test_latents': test_latents,
        'train_filtered': train_filtered_means,
        'test_filtered': test_filtered_means,
        'train_covs': train_filtered_covs,
        'test_covs': test_filtered_covs,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'kalman_filter': kf
    }

def plot_kalman_results(results, keys, output_dir, spearman_matrix=None, r2_matrix=None):
    os.makedirs(output_dir, exist_ok=True)
    
    test_latents = results['test_latents']
    test_filtered = results['test_filtered']
    test_covs = results['test_covs']
    
    n_vars = min(len(keys), test_latents.shape[1])
    
    for i in range(n_vars):
        plt.figure(figsize=(12, 6))
        plt.plot(test_latents[:, i], 'b-', label='Original')
        plt.plot(test_filtered[:, i], 'r-', label='Kalman Filter')
        
        std_devs = np.array([np.sqrt(cov[i, i]) for cov in test_covs])
        plt.fill_between(range(len(test_latents)), 
                         test_filtered[:, i] - 2 * std_devs,
                         test_filtered[:, i] + 2 * std_devs,
                         color='r', alpha=0.2)
        
        plt.title(f'Variable: {keys[i]}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"kalman_var_{keys[i]}.png"), dpi=300)
        plt.close()
    
    if spearman_matrix is not None:
        plt.figure(figsize=(10, 8))
        
        plt.imshow(np.abs(spearman_matrix), cmap='OrRd', vmin=0, vmax=0.5)
        cbar = plt.colorbar()
        cbar.set_label('Spearman\'s Rank Correlation')
        
        for i in range(len(keys)):
            for j in range(len(keys)):
                text = f"{spearman_matrix[i, j]:.2f}"
                text_color = 'white' if abs(spearman_matrix[i, j]) > 0.3 else 'black'
                plt.text(j, i, text, ha="center", va="center", 
                        color=text_color, fontsize=10)
        
        plt.xticks(range(len(keys)), keys, rotation=45)
        plt.yticks(range(len(keys)), keys)
        plt.xlabel('True causal variable')
        plt.ylabel('Target dimension')
        plt.title('Spearman\'s Rank Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "spearman_correlation_matrix.png"), dpi=300)
        plt.close()
    
    if r2_matrix is not None:
        plt.figure(figsize=(10, 8))
        plt.imshow(r2_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='R² Score')
        
        for i in range(n_vars):
            for j in range(n_vars):
                text = f"{r2_matrix[i, j]:.2f}"
                text_color = 'white' if r2_matrix[i, j] < -0.5 else 'black'
                plt.text(j, i, text, ha="center", va="center", 
                        color=text_color, fontsize=10)
        
        plt.xticks(range(n_vars), keys, rotation=45)
        plt.yticks(range(n_vars), keys)
        plt.xlabel('Test Variables')
        plt.ylabel('Filtered Variables')
        plt.title('R² Score between Original and Filtered Variables')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "r2_matrix.png"), dpi=300)
        plt.close()
    
    plt.figure(figsize=(15, 3 * n_vars))
    for i in range(n_vars):
        plt.subplot(n_vars, 1, i + 1)
        plt.plot(test_latents[:, i], 'b-', label='Original')
        plt.plot(test_filtered[:, i], 'r-', label='Kalman Filter')
        plt.title(f'Variable: {keys[i]}')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kalman_all_variables.png"), dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Kalman Filter Pong Analysis")
    parser.add_argument('--train_path', type=str, required=True, help='Path to train.npz file')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test.npz file')
    parser.add_argument('--output_dir', type=str, default='results/kalman_filter', help='Directory to save output')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for processing (None uses all data)')
    parser.add_argument('--n_iter', type=int, default=10, help='Number of EM iterations for Kalman filter')
    parser.add_argument('--transition_cov', type=float, default=0.01, help='Transition covariance scale')
    parser.add_argument('--observation_cov', type=float, default=0.1, help='Observation covariance scale')
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"batch_size_{args.batch_size}", f"kalman_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading training data from {args.train_path}")
    train_latents, train_keys = load_data(args.train_path)
    
    print(f"Loading test data from {args.test_path}")
    test_latents, test_keys = load_data(args.test_path)
    
    if args.batch_size is not None and args.batch_size < len(train_latents):
        print(f"Using {args.batch_size} samples from training data (out of {len(train_latents)})")
        train_latents = train_latents[:args.batch_size]
    
    kalman_results = apply_kalman_filter(
        train_latents, 
        test_latents,
        n_iter=args.n_iter,
        transition_cov_scale=args.transition_cov,
        observation_cov_scale=args.observation_cov
    )
    
    print("\n=== Filtered Data Metrics ===")

    train_filtered_spearman, train_filtered_r2 = compute_metrics(
        kalman_results['train_filtered'], train_latents, train_keys
    )
    print("=== TRAIN SET (FILTERED vs ORIGINAL) ===")
    print_matrix(train_filtered_spearman, train_keys, "Train Filtered Spearman Correlation Matrix")
    print_matrix(train_filtered_r2, train_keys, "Train Filtered R^2 Matrix")

    test_filtered_spearman, test_filtered_r2 = compute_metrics(
        kalman_results['test_filtered'], test_latents, test_keys
    )
    print("=== TEST SET (FILTERED vs ORIGINAL) ===")
    print_matrix(test_filtered_spearman, test_keys, "Test Filtered Spearman Correlation Matrix")
    print_matrix(test_filtered_r2, test_keys, "Test Filtered R^2 Matrix")

    train_output_dir = os.path.join(output_dir, "train_results")
    os.makedirs(train_output_dir, exist_ok=True)
    plot_kalman_results(
        {
            'test_latents': train_latents, 
            'test_filtered': kalman_results['train_filtered'], 
            'test_covs': kalman_results['train_covs']
        }, 
        train_keys,
        train_output_dir, 
        spearman_matrix=train_filtered_spearman, 
        r2_matrix=train_filtered_r2
    )

    test_output_dir = os.path.join(output_dir, "test_results")
    os.makedirs(test_output_dir, exist_ok=True)
    plot_kalman_results(
        kalman_results, 
        test_keys,
        test_output_dir, 
        spearman_matrix=test_filtered_spearman, 
        r2_matrix=test_filtered_r2
    )

    np.savez_compressed(
        os.path.join(output_dir, "kalman_results.npz"),
        train_latents=train_latents,
        test_latents=test_latents,
        train_filtered=kalman_results['train_filtered'],
        test_filtered=kalman_results['test_filtered'],
        train_rmse=kalman_results['train_rmse'],
        test_rmse=kalman_results['test_rmse'],
        train_spearman_filtered=train_filtered_spearman,
        test_spearman_filtered=test_filtered_spearman,
        train_r2_filtered=train_filtered_r2,
        test_r2_filtered=test_filtered_r2,
        keys=test_keys
    )

    # Update the summary file with both sets of metrics
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"Kalman Filter Analysis\n")
        f.write(f"Date: {timestamp}\n\n")
        f.write(f"Training data: {args.train_path}\n")
        f.write(f"Test data: {args.test_path}\n")
        f.write(f"Number of variables: {len(test_keys)}\n")
        f.write(f"Training samples: {len(train_latents)}\n")
        f.write(f"Test samples: {len(test_latents)}\n")
        f.write(f"Kalman EM iterations: {args.n_iter}\n\n")
        f.write(f"Training RMSE: {kalman_results['train_rmse']:.4f}\n")
        f.write(f"Test RMSE: {kalman_results['test_rmse']:.4f}\n\n")
        f.write(f"=== Train Filtered vs Original Metrics ===\n")
        f.write(f"Average absolute Spearman correlation: {np.nanmean(np.abs(train_filtered_spearman)):.4f}\n")
        f.write(f"Average R² score: {np.nanmean(train_filtered_r2):.4f}\n\n")
        f.write(f"=== Test Filtered vs Original Metrics ===\n")
        f.write(f"Average absolute Spearman correlation: {np.nanmean(np.abs(test_filtered_spearman)):.4f}\n")
        f.write(f"Average R² score: {np.nanmean(test_filtered_r2):.4f}\n")

    print(f"\nAll results saved to {output_dir}")
    print(f"- Train results: {train_output_dir}")
    print(f"- Test results: {test_output_dir}")