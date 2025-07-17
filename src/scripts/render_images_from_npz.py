"""
Render images from an NPZ file, handling both individual images and triplets.
"""
import argparse
import numpy as np
from matplotlib import pyplot as plt
import os

def render_npz_images(npz_file, save_folder=None, display=True, num_samples=None):
    print(f"Loading NPZ file: {npz_file}")
    data = np.load(npz_file, allow_pickle=True)
    
    # Adjust image keys based on npz file structure
    image_keys = [key for key in data.files if key in ['images', 'predicted_next', 'data']]
    
    if not image_keys:
        print("No image data found. Available keys:", list(data.files))
        return
        
    image_key = image_keys[0]
    images = data[image_key]
    
    print(f"Image data shape: {images.shape}")
    
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
    
    # Handling based on dimensionality
    if len(images.shape) == 3:  # Single image: (height, width, channels)
        images = images[np.newaxis, ...]  # Add a sample dimension
        process_regular_images(images, save_folder, display, num_samples)
    elif len(images.shape) == 4:  # (samples, height, width, channels)
        process_regular_images(images, save_folder, display, num_samples)
    elif len(images.shape) == 5:  # Triplets: (samples, triplet, height, width, channels)
        process_triplet_images(images, save_folder, display, num_samples)
    elif len(images.shape) == 2:  # (samples, width)
        process_1d_images(images, save_folder, display, num_samples)
    else:
        print(f"Unexpected image data shape: {images.shape}")


def process_1d_images(images, save_folder, display, num_samples):
    N, W = images.shape
    H = W

    rgb = np.zeros((N, H, W, 3), dtype=np.uint8)
    alpha = np.repeat(images[:, np.newaxis, :], H, axis=1)
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    alpha = alpha[..., np.newaxis]  # (N, H, W, 1)

    images = np.concatenate([rgb, alpha], axis=-1)  # (N, H, W, 4)

    for i in range(N):
        plt.figure(figsize=(4, 4))
        plt.imshow(images[i].astype(np.uint8))
        plt.axis('off')
        plt.title(f"Image {i}")
        
        if save_folder:
            plt.savefig(os.path.join(save_folder, f"image_{i}.png"), 
                       bbox_inches='tight', pad_inches=0.1)
        
        if display:
            plt.show()
        else:
            plt.close()


def process_regular_images(images, save_folder, display, num_samples):
    total = images.shape[0] if num_samples is None else min(num_samples, images.shape[0])
    
    for i in range(total):
        plt.figure(figsize=(4, 4))
        plt.imshow(images[i].astype(np.uint8))
        plt.axis('off')
        plt.title(f"Image {i}")
        
        if save_folder:
            plt.savefig(os.path.join(save_folder, f"image_{i}.png"), 
                       bbox_inches='tight', pad_inches=0.1)
        
        if display:
            plt.show()
        else:
            plt.close()


def process_triplet_images(images, save_folder, display, num_samples):
    total = images.shape[0] if num_samples is None else min(num_samples, images.shape[0])
    
    for i in range(total):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        for j in range(3):
            axes[j].imshow(images[i, j].astype(np.uint8))
            axes[j].axis('off')
            axes[j].set_title(f"Frame {j}")
        
        plt.suptitle(f"Triplet {i}")
        plt.tight_layout()
        
        if save_folder:
            plt.savefig(os.path.join(save_folder, f"triplet_{i}.png"), 
                       bbox_inches='tight', pad_inches=0.1)
            
            for j in range(3):
                plt.figure(figsize=(4, 4))
                plt.imshow(images[i, j].astype(np.uint8))
                plt.axis('off') 
                plt.savefig(os.path.join(save_folder, f"triplet_{i}_frame_{j}.png"),
                           bbox_inches='tight', pad_inches=0)
                plt.close()
        
        if display:
            plt.show()
        else:
            plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Open and render images from an NPZ file.")
    parser.add_argument('npz_file', type=str, help='Path to the NPZ file to open')
    parser.add_argument('--save', type=str, default=None, 
                        help='Folder to save images to (optional)')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display images, just save them')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to process (default: all)')
    
    args = parser.parse_args()
    
    render_npz_images(
        args.npz_file,
        save_folder=args.save,
        display=not args.no_display,
        num_samples=args.num_samples
    )