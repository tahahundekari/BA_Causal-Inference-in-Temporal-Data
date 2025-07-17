"""
This script computes the errors from Kalman and Citris error images (differences between test and prediction images).
"""
import numpy as np


def compute_errors(
    error_images_kalman_npz_path: str,
    error_images_citris_npz_path: str,
    output_npz_path: str = "errors.npz"
):
    kalman_error_images = np.load(error_images_kalman_npz_path)["error_images"]
    citris_error_images = np.load(error_images_citris_npz_path)["images"]

    min_length = min(kalman_error_images.shape[0], citris_error_images.shape[0]) 

    kalman_error_images = kalman_error_images[:min_length]
    citris_error_images = citris_error_images[:min_length]
    
    mean_rows = citris_error_images.mean(axis=1)
    citris_error_images_corrected = np.repeat(mean_rows[:, np.newaxis, :, :], citris_error_images.shape[1], axis=1)

    citris_error_images_corrected[..., :3] = 0
    citris_error_images_corrected = citris_error_images_corrected[:, 0, :, :]
    citris_error_images_corrected = citris_error_images_corrected[..., 3]

    kalman_errors = kalman_error_images.mean(axis=1)
    citris_errors = citris_error_images_corrected.mean(axis=1)

    kalman_average_error = np.mean(kalman_errors)
    citris_average_error = np.mean(citris_errors)

    np.savez(
        output_npz_path,
        kalman_errors=kalman_errors,
        citris_errors=citris_errors,
        kalman_average_error=kalman_average_error,
        citris_average_error=citris_average_error,
    )



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute errors from Kalman and Citris error images NPZ files.")
    parser.add_argument('error_images_kalman_npz_path', type=str, help='Path to the Kalman error images NPZ file')
    parser.add_argument('error_images_citris_npz_path', type=str, help='Path to the Citris error images NPZ file')
    parser.add_argument('--output_npz_path', type=str, default='errors.npz', help='Path to save the computed errors NPZ file')

    args = parser.parse_args()

    compute_errors(args.error_images_kalman_npz_path, args.error_images_citris_npz_path, args.output_npz_path)