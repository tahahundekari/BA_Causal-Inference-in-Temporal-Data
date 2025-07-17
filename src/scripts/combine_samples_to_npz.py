"""
This script combines the .npy files of the citris predictions into a single .npz file.
"""

def combine_samples_to_npz(input_dir):
    import os
    import numpy as np

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory {input_dir} does not exist or is not a directory.")

    sub_directories = [f for f in os.listdir(input_dir) if f.startswith('sample_0_') and os.path.isdir(os.path.join(input_dir, f))]

    originals_data_list = []
    predictions_data_list = []

    for sub_directory in sub_directories:
        original_file =  os.path.join(input_dir, sub_directory, "original.npy")
        original_file_path = os.path.join(input_dir, original_file)
        original_data = np.load(original_file_path)
        original_data = original_data.tolist() if isinstance(original_data, np.ndarray) else original_data
        originals_data_list.append([original_data])

        prediction_file = os.path.join(input_dir, sub_directory, "prediction.npy")
        prediction_file_path = os.path.join(input_dir, prediction_file)
        prediction_data = np.load(prediction_file_path)
        prediction_data = prediction_data.tolist() if isinstance(prediction_data, np.ndarray) else prediction_data
        predictions_data_list.append([prediction_data])

    combined_original_data = np.concatenate(originals_data_list, axis=0)
    combined_prediction_data = np.concatenate(predictions_data_list, axis=0)

    np.savez(os.path.join(input_dir, "originals_sample_0.npz"), images=combined_original_data)
    np.savez(os.path.join(input_dir, "predictions_sample_0.npz"), images=combined_prediction_data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine multiple .npy files into a single .npz file.")
    parser.add_argument("input_dir", type=str, help="Directory containing .npy files to combine.")

    args = parser.parse_args()

    combine_samples_to_npz(args.input_dir)
    print(f"Combined samples from {args.input_dir} into {args.input_dir}/originals_sample_0.npz and {args.input_dir}/predictions_sample_0.npz.")