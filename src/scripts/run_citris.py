import os
import sys

sys.path.append('../..')
from citris.models.citris_vae import CITRISVAE

def run_citris_on_dataset(data_path, epochs=100, batch_size=64, learning_rate=1e-3):
    # load dataset

    pass

if __name__ == '__main__':
    data_dir = os.path.join("data", "raw", "structured_linear")
    run_citris_on_dataset(os.path.join(data_dir, "structured_latent_observations.csv"))
    print("Run CITRIS-VAE training script")