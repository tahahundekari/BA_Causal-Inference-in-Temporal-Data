## Causal Inference in Temporal Data

#### This bachelor's thesis compares Causal Representation Learning (CRL) in temporal data with time-series forecasting with Kalman Filters.

### File Structure
- data
    - raw -> Datasets used by CITRIS and iCITRIS
    - process -> Preprocessed times series
    - external -> External (real-world) data sets

### Setup
- Install Anaconda or Miniconda [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Create and activate conda environment using the provided environment.yml file
```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate ba-causal-inference-in-temporal-data
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