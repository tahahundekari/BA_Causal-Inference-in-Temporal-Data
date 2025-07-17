"""
Plot the Kalman and CITRIS errors.
"""
import numpy as np
import matplotlib.pyplot as plt

def generate_error_plots(error_path: str, T: int = 32, output_path: str = 'error_comparison_plot.png'):
    errors = np.load(error_path)

    kalman_errors = errors['kalman_errors'][:T]
    citris_errors = errors['citris_errors'][:T]

    plt.figure(figsize=(10, 5))
    plt.plot(kalman_errors, label='Kalman Errors', color='blue', linewidth=2)
    plt.plot(citris_errors, label='CITRIS Errors', color='orange', linewidth=2)
    plt.title('Error Comparison: Kalman vs CITRIS')
    plt.xlabel('Sample Index')
    plt.ylabel('Error Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate error comparison plots from NPZ file.")
    parser.add_argument('error_path', type=str, help='Path to the NPZ file containing errors')
    parser.add_argument('--T', type=int, default=32, help='Number of samples to plot (default: 32)')
    parser.add_argument('--output_path', type=str, default='error_comparison_plot.png', help='Path to save the error comparison plot')

    args = parser.parse_args()

    generate_error_plots(args.error_path, args.T, args.output_path)
