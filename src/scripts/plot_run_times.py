"""
Plot the run times over batch-sizes of Kalman and CITRIS methods.
"""
import math
import matplotlib.pyplot as plt

def plot_run_times(output_path: str = 'run_times_plot.png'):
    kalman_run_time_10000 = 33.301220
    kalman_run_time_25000 = 75.127759
    kalman_run_time_50000 = 146.244626

    citris_run_time_10000 = 6160.204964
    citris_run_time_25000 = 7003.998900
    citris_run_time_50000 = 7847.346572

    plt.figure(figsize=(10, 5))
    plt.plot([10000, 25000, 50000], [math.log(kalman_run_time_10000), math.log(kalman_run_time_25000), math.log(kalman_run_time_50000)], 'ro', label='Kalman Times', color='blue', markersize=8)
    plt.plot([10000, 25000, 50000], [math.log(citris_run_time_10000), math.log(citris_run_time_25000), math.log(citris_run_time_50000)], 'ro', label='CITRIS Times', color='orange', markersize=8)
    plt.title('Run Time Comparison: Kalman vs CITRIS')
    plt.xlabel('Batch Size')
    plt.ylabel('Log Run Time (seconds)')
    plt.legend(loc='center right', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot run times of different methods.")
    parser.add_argument('--output_path', type=str, default='run_times_plot.png', help='Path to save the plot')
    
    args = parser.parse_args()
    
    plot_run_times(args.output_path)