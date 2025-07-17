"""
This script computes the CITRIS error images from test and predicted images stored in NPZ files.
"""
def compute_error_images_citris(
    originals_npz_path: str,
    predictions_npz_path: str,
    output_npz_path: str,
):
    import numpy as np

    originals_data = np.load(originals_npz_path)
    predictions_data = np.load(predictions_npz_path)

    originals_images = originals_data['images']
    predictions_images = predictions_data['images']

    if originals_images.shape != predictions_images.shape:
        raise ValueError("Originals and predictions must have the same shape.")

    error_images = np.abs(originals_images - predictions_images)

    np.savez(output_npz_path, images=error_images)
    print(f"Error images saved to {output_npz_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute error images from originals and predictions NPZ files.")
    parser.add_argument('originals_npz_path', type=str, help='Path to the originals NPZ file')
    parser.add_argument('predictions_npz_path', type=str, help='Path to the predictions NPZ file')
    parser.add_argument('output_npz_path', type=str, help='Path to save the computed error images NPZ file')

    args = parser.parse_args()

    compute_error_images_citris(args.originals_npz_path, args.predictions_npz_path, args.output_npz_path)