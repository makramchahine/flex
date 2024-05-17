# Flex: Fly lexical

## General

### Setup
```
$ conda create -n flex python==3.9
$ pip install ipdb
# install torch; check https://pytorch.org/get-started/locally/
```

## Learning

### Setup
Installation.
```
$ pip install hydra-core==1.3.2
$ pip install hydra_colorlog --upgrade
$ cd learning
$ pip install -e .
$ pip install tensorboard
$ pip install cvxopt
$ pip install pyrootutils
$ pip install salesforce-lavis # for BLIP
$ pip install vit-pytorch==1.2.4 # transformer-based policy
```

copy learning/src/models/components/extractors/eva_vit.py to installation lavis/model/eva_vit.py (probably @ ~/miniconda3/envs/fm_flight/lib/python3.9/site-packages/lavis/models/eva_vit.py)


Setup paths.
```
$ export DATA_DIR=<directory-of-dataset>
$ export LOG_DIR=<directory-to-store-outputs>
```
You can also edit and run `scripts/setup_macro.sh`.

### Workflow
* Go to `learning/`.
* Develop datamodule with testing script in `scripts/test_datamodule.sh`.
* Develop model with testing script in `scripts/test_model.sh`.
* Develop module with testing script in `scripts/test_module.sh`.
* Set `HYDRA_FULL_ERROR=1` to get traceback for debugging.

### Run
Training.
```
$ bash scripts/train_flight.sh # you can specify directory name in bash arguments
```
