"""
Preprocess Pong images by replacing ball with black columns.
"""
import numpy as np

def squash_ball_to_black_columns(data):
    images = data["images"]

    grey=(0, 0, 0, 42)

    if images.ndim == 4:  # (N, H, W, C)
        N, H, W, C = images.shape
        out_images = np.zeros((N, H, W, C), dtype=np.uint8)
        grey = (0, 0, 0, 42)

        for i in range(N):
            img = images[i]
            mask_ball = (img[..., 0] > 200) & (img[..., 1] < 80) & (img[..., 2] < 80) & (img[..., 3] > 128)
            cols_with_ball = mask_ball.any(axis=0)

            cols_with_ball_indices = np.where(cols_with_ball)[0] 
            if cols_with_ball_indices.size == 1 and cols_with_ball_indices[0] < 31:
                cols_with_ball_indices = np.array([cols_with_ball_indices[0], cols_with_ball_indices[0] + 1])
            elif cols_with_ball_indices.size == 1 and cols_with_ball_indices[0] >= 31:
                cols_with_ball_indices = np.array([cols_with_ball_indices[0] - 1, cols_with_ball_indices[0]])
            elif cols_with_ball_indices.size == 3:
                cols_with_ball_indices = np.array([cols_with_ball_indices[0], cols_with_ball_indices[1]])

            cols_with_ball[cols_with_ball_indices[0]] = True
            cols_with_ball[cols_with_ball_indices[1]] = True

            row = np.tile(grey, (W, 1))
            row[cols_with_ball] = [0, 0, 0, 255]

            out_img = np.tile(row[None, :, :], (H, 1, 1))
            out_images[i] = out_img

    elif images.ndim == 5:  # (N, triplet, H, W, C)
        N, T, H, W, C = images.shape
        out_images = np.zeros((N, T, H, W, C), dtype=np.uint8)
        grey = (0, 0, 0, 42)

        for i in range(N):
            for t in range(T):
                img = images[i, t]
                mask_ball = (img[..., 0] > 200) & (img[..., 1] < 80) & (img[..., 2] < 80) & (img[..., 3] > 128)
                cols_with_ball = mask_ball.any(axis=0)

                cols_with_ball_indices = np.where(cols_with_ball)[0] 
                if cols_with_ball_indices.size == 1 and cols_with_ball_indices[0] < 31:
                    cols_with_ball_indices = np.array([cols_with_ball_indices[0], cols_with_ball_indices[0] + 1])
                elif cols_with_ball_indices.size == 1 and cols_with_ball_indices[0] >= 31:
                    cols_with_ball_indices = np.array([cols_with_ball_indices[0] - 1, cols_with_ball_indices[0]])
                elif cols_with_ball_indices.size == 3:
                    cols_with_ball_indices = np.array([cols_with_ball_indices[0], cols_with_ball_indices[1]])

                cols_with_ball[cols_with_ball_indices[0]] = True
                cols_with_ball[cols_with_ball_indices[1]] = True

                row = np.tile(grey, (W, 1))
                row[cols_with_ball] = [0, 0, 0, 255]

                out_img = np.tile(row[None, :, :], (H, 1, 1))
                out_images[i, t] = out_img
    else:
        raise ValueError(f"Unsupported image shape with {images.ndim} dimensions")

    result = {key: data[key] for key in data.files}
    result["images"] = out_images

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Squash images by replacing columns with black balls with black.")
    parser.add_argument('input_file', type=str, help='Path to the input .npz file')
    parser.add_argument('output_file', type=str, help='Path to save the squashed .npz file')
    
    args = parser.parse_args()

    data = np.load(args.input_file)

    updated_data = squash_ball_to_black_columns(data)

    np.savez_compressed(args.output_file, **updated_data)
    print(f"Saved squashed images to: {args.output_file}")
