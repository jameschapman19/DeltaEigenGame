# Unconstrained Stochastic CCA: Unifying Multiview and Self-Supervised Learning

This repository contains the code for the paper "Stochastic Gradient Descent Algorithms for Generalized Eigenvalue Problems and Self-Supervised Learning"

We use wandb to log the experiments. To run the experiments, you need to set up a wandb account and set the environment variable `WANDB_API_KEY` to your API key. You can find your API key in your account settings.

## Data

The data for the experiments can be downloaded by following the instructions from https://github.com/zihangm/riemannian-streaming-cca. The data should be placed in the `data` folder
with the following structure:

```
data
├── CIFAR
│   ├── test_x.mat
│   └── train_x.mat
│   ├── test_y.mat
│   └── train_y.mat
├── MNIST
│   ├── mnist.mat
├── Mediamill
│   ├── mediamill_trainX.mat
│   └── mediamill_trainY.mat
```

## Requirements

The code is written in Python 3.9 and requires the packages in `requirements.txt`. To install them, run

```
pip install -r requirements.txt
```

## Experiments

### Stochastic CCA

To run the experiments for stochastic CCA, run

```
python -m train.py
```

To change the arguments, either change the default values in `train.py` or pass them as command line arguments. For example, to run the experiments with a batch size of 100, run

```
python -m train.py --batch_size 100
```

### Deep CCA

To run the experiments for deep CCA, run

```
python -m deep_train.py
```

To change the arguments, either change the default values in `deep_train.py` or pass them as command line arguments. For example, to run the experiments with a batch size of 100, run

```
python -m deep_train.py --batch_size 100
```
