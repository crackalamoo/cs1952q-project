# CSCI 1952Q Final Project
## Training Neural Networks with Adaptive Subset Selection
Deep Neural Networks (DNNs) are pivotal in advancing various disciplines but are hindered by high data and computational demands. This is the code for our paper on "Adaptive Subset Selection," a technique designed to improve training efficiency by selectively employing data subsets that are most beneficial for training. This method, applied to the MNIST dataset using a convolutional neural network (CNN), maintained comparable accuracy (within 1\% of full dataset training) and cut training time by 40\%. This approach demonstrates potential for reducing computational requirements without substantially compromising accuracy, providing significant insights for efficient DNN training.

## Running
* To download the MNIST dataset in our format, you may run `cd code` and `python preprocess.py --dataset mnist`.
* To train a standard CNN on this dataset, you may run `python main.py`. Make sure to remain in the `code` folder. Note that this runs three times with different seeds by default; this can be adjusted with the `--runs` argument.
* Next, to train a model with adaptive subset selection, you may run `python main.py --samp`.

## Parameters
* `--samp`: boolean, whether to use adaptive subset selection. Default: `False`.
* `--device`: device used for PyTorch training. Default: `cpu`.
* `--epochs`: number of epochs to train for. Default: `12`.
* `--model`: which model to use. Default: `mnist`. This is the only model we have extensively developed, but `wmt` (machine translation) and `cifar10` (10-class color image classification) are also available.
* `--runs`: number of runs with different seeds over which to save data. Default: `3`.
* `--samp-prop`: proportion of training samples with the highest loss to include with adaptive subset selection. Hyperparameter $p$ in our paper. Default: `0.35`.
* `--reintroduce`: proportion of normally excluded data to randomly reintroduce with adaptive subset selection. Hyperparameter $\alpha$ in our paper. Default: `0.20`.
* `--store-loss`: number of losses to store with adaptive subset selection. Hyperparameter $N$ in our paper. Default: `2`.