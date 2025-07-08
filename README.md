## Causal Inference in Temporal Data

#### This bachelor's thesis compares Causal Representation Learning (CRL) in temporal data with time-series forecasting with Kalman Filters.

### File Structure
- `data`
    - `raw` -> Datasets used by CITRIS
    - `process` -> Preprocessed times series

- `citris` -> All scripts written by CITRIS authors

- `src/scripts` -> All scripts not relating to CITRIS explicitly
    

### Setup
- Install Anaconda or Miniconda [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Create and activate conda environment using the provided environment.yml file
```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate ba-causal-inference-in-temporal-data
```

## Generating the data
The interventional pong dataset can be found under [this repository]("https://github.com/phlippe/CITRIS"), which we will save under `data/raw/interventional_pong` to be used for further processing. First, a copy of this should be made in `data/processed/` with a particular name `<copy_name>` and the following commands should be run.

```bash
python src/scripts/squash_images_greyscale_in_npz.py \
data/raw/interventional_pong/train.npz \
data/processed/<copy_name>/train.npz

python src/scripts/squash_images_greyscale_in_npz.py \
data/raw/interventional_pong/test.npz \
data/processed/<copy_name>/test.npz

python src/scripts/squash_images_greyscale_in_npz.py \
data/raw/interventional_pong/test_indep.npz \
data/processed/<copy_name>/test_indep.npz

python src/scripts/squash_images_greyscale_in_npz.py \
data/raw/interventional_pong/test_triplets.npz \
data/processed/<copy_name>/test_triplets.npz
```


## Running the training scripts

### Kalman Filter
```bash
cd src/scripts

# For batch-size of 10000 time-steps
python run_kalman_pong.py \
  --train_path data/processed/<copy_name>/train_kalman.npz \
  --test_path data/processed/<copy_name>/test_kalman.npz \
  --output_dir results/kalman_pong_preprocessed \
  --batch_size 10000

# For batch-size of 25000 time-steps
python run_kalman_pong.py \
  --train_path data/processed/<copy_name>/train_kalman.npz \
  --test_path data/processed/<copy_name>/test_kalman.npz \
  --output_dir results/kalman_pong_preprocessed \
  --batch_size 25000

# For batch-size of 50000 time-steps
python run_kalman_pong.py \
  --train_path data/processed/<copy_name>/train_kalman.npz \
  --test_path data/processed/<copy_name>/test_kalman.npz \
  --output_dir results/kalman_pong_preprocessed \
  --batch_size 50000
```

### CITRIS
```bash
cd citris

# Training the Causal Encoder
python experiments/train_causal_encoder.py \
--data_dir ../data/processed/<copy_name> \
--max_epochs 10 \
--num_workers 4

# Training CITRIS VAE
# For batch-size of 10000 time-steps
python experiments/train_vae.py \
--data_dir ../data/processed/<copy_name> \
--causal_encoder_checkpoint checkpoints/CausalEncoder/<Causal_Encoder_name>/version_<version_number>/checkpoints/<checkpoint>.ckpt \
--batch_size 10000 \
--num_workers 4 \
--max_epochs 10

# For batch-size of 25000 time-steps
python experiments/train_vae.py \
--data_dir ../data/processed/<copy_name> \
--causal_encoder_checkpoint checkpoints/CausalEncoder/<Causal_Encoder_name>/version_<version_number>/checkpoints/<checkpoint>.ckpt \
--batch_size 25000 \
--num_workers 4 \
--max_epochs 10

# For batch-size of 50000 time-steps
python experiments/train_vae.py \
--data_dir ../data/processed/<copy_name> \
--causal_encoder_checkpoint checkpoints/CausalEncoder/<Causal_Encoder_name>/version_<version_number>/checkpoints/<checkpoint>.ckpt \
--batch_size 50000 \
--num_workers 4 \
--max_epochs 10
```

### CITRIS and iCITRIS reference
```bibtex
@inproceedings{lippe2022citris,
   title        = {{CITRIS}: Causal Identifiability from Temporal Intervened Sequences},
   author       = {Lippe, Phillip and Magliacane, Sara and L{\"o}we, Sindy and Asano, Yuki M and Cohen, Taco and Gavves, Stratis},
   year         = {2022},
   month        = {17--23 Jul},
   booktitle    = {Proceedings of the 39th International Conference on Machine Learning},
   publisher    = {PMLR},
   series       = {Proceedings of Machine Learning Research},
   volume       = {162},
   pages        = {13557--13603},
   url          = {https://proceedings.mlr.press/v162/lippe22a.html},
   editor       = {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan}
}
```
```bibtex
@inproceedings{lippe2023causal,
    title        = {Causal Representation Learning for Instantaneous and Temporal Effects in Interactive Systems},
    author       = {Phillip Lippe and Sara Magliacane and Sindy L{\"o}we and Yuki M Asano and Taco Cohen and Efstratios Gavves},
    year         = 2023,
    booktitle    = {The Eleventh International Conference on Learning Representations},
    url          = {https://openreview.net/forum?id=itZ6ggvMnzS}
}
```