Public repository for neuron embedding paper.

# Representing Individual Neurons for Neuroevolution

This repository is the official implementation of Representing Individual Neurons for Neuroevolution.

## Requirements

This code was tested on Python 3.8.

To install requirements:

```setup
pip install -r requirements.txt
```

Visdom should be running in order to view the plots.

## Training

To train the models in the paper, run this command:

```train
python run_train_test.py -m <modeltype> -e <epochs> -d <embedding dimension>
python run_train_test.py -c <configname>
```

```configname``` is the name of the configuration found in ```configs/settings.conf```. All training settings should be changed there.

Example:
```
python run_train_test.py -c Emb
```

```<modeltype>``` can be one of FC (fully connected, direct encoding), Conv (convolutional, direct encoding), Emb (fully connected, neuron embedding), EmbConv (convolutional, neuron embedding).

After every epoch, training model will be saved as ```epoch<epochnum>.pth``` in the ```models``` folder.


## Evaluation
Results from evaluation are displayed in Visdom. Ensure Visdom is open before running.

### Evaluating performance
To evaluate a trained model, run:
```
python run_train_test.py -m <modeltype>  -d <embedding dimension> -l <model to load>
```
Example:
```
python run_train_test.py -m Emb  -d 64 -l models/trained_to_ref_1.pth
```

### Matching reference
To train a model to match a reference, run:
```eval
python run_train_to_ref.py -r <target fully connected model> -s <path to save model> -c <fully connected config setting> -e <embedded model config>
```
Example:
```
python run_train_to_ref.py -r models/fc.pth -s models/tmp.pth -c FC -e Emb_ref
```

### Interpolation

To run the crossover experiment, run:
```eval
python run_interpolation.py
```

Settings for this experiment are in ```crossover.conf```.

### Crossover

To run the crossover experiment, run:
```eval
python run_crossover.py -l <layer number>
```
Example:
```eval
python run_crossover.py -l 1
```

Settings for this experiment are in ```crossover.conf```.


## Pre-trained Models

Pretrained models are included in the models folder.

```
fc.pth  - first fully connected model, trained from random initialization
fc2.pth - second fully connected model with same settings, different random seed
mnist_conv_embedding_32dim.pth - 32 dimensional convolutional embedding model
mnist_conv_embedding_64dim.pth - 64 dimensional convolutional embedding model
mnist_neuron_embedding_64dim.pth - fully connected neuron embedding model, trained from random initialization
trained_to_ref_1.pth - fully connected neuron embedding model, trained to match weights of fc.pth
trained_to_ref_2.pth - fully connected neuron embedding model, trained to match weights of fc2.pth
```
