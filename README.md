Public repository for neuron embedding paper.

# Representing Individual Neurons for Neuroevolution

This repository is the official implementation of Representing Individual Neurons for Neuroevolution.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Visdom should be running in order to view the plots.

## Training

To train the models in the paper, run this command:

```train
python run_train_test.py -m <modeltype> -e <epochs> -d <embedding dimension>
```

Example:
```
python run_train_test.py -m Emb -e 2 -d 64
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
python run_train_to_ref.py -r <target fully connected model> -s <path to save model> -d <embedding dimension>          
```
Example:
```
python run_train_to_ref.py -r models/fc.pth -s models/tmp.pth -d 64
```

### Crossover

To run the crossover experiments, run:
```eval
python run_crossover.py -fc1 <recipient FC model>  -fc2 <donor FC model> -emb1 <recipient embedding model> -emb2 <donor embedding model>
```
Example:
```eval
python run_crossover.py -fc1 models/fc.pth -fc2 models/fc2.pth -emb1 models/trained_to_ref_1.pth -emb2 models/trained_to_ref_2.pth
```

Use the parameter ```-c``` to select linear interpolation (1) or neuron transplant (2).

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
