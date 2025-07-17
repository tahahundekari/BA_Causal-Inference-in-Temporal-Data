"""
Extract the alpha channel from images for the Kalman filter input.
"""
def process_images_for_kalman(images):
    if images.ndim == 4:  # (N, H, W, C)
        processed_images = images[:, 0, :, 3]  # Extract the alpha channel
    else:
        raise ValueError(f"Unsupported image shape with {images.ndim} dimensions")
    
    return processed_images


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Process images for Kalman filter input.")
    parser.add_argument('input_file', type=str, help='Path to the input .npz file')
    parser.add_argument('output_file', type=str, help='Path to save the processed .npz file')

    args = parser.parse_args()

    data = np.load(args.input_file, allow_pickle=True)
    
    processed_images = process_images_for_kalman(data['images'])

    np.savez(args.output_file, images=processed_images)
    print(f"Processed images saved to {args.output_file}")