import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np

import visdom
import matplotlib.pyplot as plt

import time
import models
import config
from traintest import *
from util import getdev, load
import loader
from models import scaled_dot, unscaled_dot, get_embeddings, get_layers, get_weights

from torchsummary import summary

import embedding
import argparse

def interpolate(model1, model2, coefficient):
    '''
    Does linear interpolation between the weights of two models.
    '''

    # seems like it's safer to do it via the state_dict?
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    interp_dict = {k: torch.lerp(sd1[k], sd2[k], coefficient) for k in sd1.keys()}
    return interp_dict

def swap_neurons(model1, model2, layer_names, neuron_ids):
    '''
    Swaps neurons specified by neuron_ids in layer_name from model2 to model1.
    
    layer_names: list of layers to swap
    n_neurons: list of lists of neurons to swap in each layer, in the same order as layer_names
    '''
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()
    
    sd_new = sd1.copy()

    
    for layer_name, ids in zip(layer_names, neuron_ids):
        sd_new[layer_name][ids] = sd2[layer_name][ids].clone().detach()
#         sd1[layer_name][:n] = sd2[layer_name][:n]
    return sd_new

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-fc1", type=str, help="path to first/recipient fully connected model", required=True)
    parser.add_argument("-fc2", type=str, help="path to second/donor fully connected model", required=True)
    parser.add_argument("-emb1", type=str, help="path to first/recipient fully connected model", required=True)
    parser.add_argument("-emb2", type=str, help="path to second/donor fully connected model", required=True)
    parser.add_argument("--crossmode", "-c", type=int, help="Crossover mode: 1 - linear interpolation, 2 - neuron transplant", default=1)
    parser.add_argument("--dims", "-d", type=int, help="number of dimensions in embedding", default=64)
    args = parser.parse_args()


    dev = getdev()

    config.vis = visdom.Visdom()
    vis = config.vis

    dataset = 'MNIST'

    if dataset == 'MNIST':
        train_loader, val_loader, test_loader = loader.MNIST()
        input_size = (1, 28, 28)
    elif dataset == 'CIFAR10':
        train_loader, val_loader, test_loader = loader.MNIST()
        input_size = (3, 32, 32)

    ref_model_1 = models.FCNet().to(dev)
    ref_model_2 = models.FCNet().to(dev)
    emb_model_1 = models.NeuronEmbedding(args.dims).to(dev)
    emb_model_2 = models.NeuronEmbedding(args.dims).to(dev)

    load(ref_model_1, args.fc1)
    load(ref_model_2, args.fc2)
    load(emb_model_1, args.emb1)
    load(emb_model_2, args.emb2)

    # interpolation models
    ref_model_lerp = models.FCNet().to(dev)
    emb_model_lerp = models.NeuronEmbedding(args.dims).to(dev)

    if args.crossmode == 1:
        # do linear interpolation
        # linear interpolation of FC
        for coeff in np.linspace(0, 1, 21):
            # linear interpolation of FC model
            ref_model_lerp.load_state_dict(interpolate(ref_model_1, ref_model_2, coeff))
            loss_test = evaluate(ref_model_lerp, test_loader, test_metrics).flatten()
            
            vis.line(X=np.array([coeff]), Y=np.array([loss_test[0]]), win='interpolation loss', name='Direct Encoding', update = 'append')
            vis.line(X=np.array([coeff]), Y=np.array([loss_test[1]]), win='interpolation acc', name='Direct Encoding', update = 'append')

            log(f'Coefficient {coeff:.3f}: val loss: {loss_test[0]:.3f}, val acc: {loss_test[1]:.3f}', vis)

        # linear interpolation of embedding model
        for coeff in np.linspace(0, 1, 21):
            emb_model_lerp.load_state_dict(interpolate(emb_model_1, emb_model_2, coeff))
            loss_test = evaluate(emb_model_lerp, test_loader, test_metrics).flatten()
            
            vis.line(X=np.array([coeff]), Y=np.array([loss_test[0]]), win='interpolation loss', name='Neuron Embedding', update = 'append')
            vis.line(X=np.array([coeff]), Y=np.array([loss_test[1]]), win='interpolation acc', name='Neuron Embedding', update = 'append')

            log(f'Coefficient {coeff:.3f}: val loss: {loss_test[0]:.3f}, val acc: {loss_test[1]:.3f}', vis)


        # make the graphs look nice
        vis.update_window_opts(
            win='interpolation loss',
            opts=dict(
                showlegend = True,
                title = 'Crossentropy loss, linear interpolation',
                xlabel = 'Interpolation coefficient',
                ylabel = 'Crossentropy loss',
                width = 600,
                height = 400
            )
        )

        vis.update_window_opts(
            win='interpolation acc',
            opts=dict(
                showlegend = True,
                title = 'Accuracy, linear interpolation',
                xlabel= 'Interpolation coefficient',
                ylabel= 'Accuracy',
                width= 600,
                height= 400,
                ytickmin=0,
                ytickmax=1
            ),
        )

    elif args.crossmode == 2:
        # do neuron crossover

        # set order of transplant
        # can be used to test a different order
        ids_1 = np.arange(ref_model_1.layer_size[1]) #np.random.permutation(ref_model_1.layer_size[1])
        ids_2 = np.arange(ref_model_1.layer_size[2]) #np.random.permutation(ref_model_1.layer_size[2])

        # neuron swapping FC models
        for coeff in np.linspace(0, 1, 21):
            n_layer_1, n_layer_2 = round(ref_model_1.layer_size[1] * coeff), round(ref_model_1.layer_size[2] * coeff)
            
            ref_model_lerp.load_state_dict(swap_neurons(ref_model_1, ref_model_2,
                                                        layer_names = ['fc1.weight','fc1.bias'],
                                                        neuron_ids = [ids_1[:n_layer_1], ids_1[:n_layer_1]]))
            loss_test = evaluate(ref_model_lerp, test_loader, test_metrics).flatten()
            
            vis.line(X=np.array([coeff]), Y=np.array([loss_test[0]]), win='crossover loss', name='Direct Encoding', update = 'append')
            vis.line(X=np.array([coeff]), Y=np.array([loss_test[1]]), win='crossover acc', name='Direct Encoding', update = 'append')

            log(f'Coefficient {coeff:.3f}: val loss: {loss_test[0]:.3f}, val acc: {loss_test[1]:.3f}', vis)


        # neuron swapping for embeddings
        for coeff in np.linspace(0, 1, 21):
            n_layer_1, n_layer_2 = round(emb_model_1.layer_size[1] * coeff), round(emb_model_1.layer_size[2] * coeff)
            emb_model_lerp.load_state_dict(swap_neurons(emb_model_1, emb_model_2,
                                                        layer_names = ['embs.1.weight', 'biases.0'],
                                                        neuron_ids = [ids_1[:n_layer_1], ids_1[:n_layer_1]]))
            loss_test = evaluate(emb_model_lerp, test_loader, test_metrics).flatten()
            
            vis.line(X=np.array([coeff]), Y=np.array([loss_test[0]]), win='crossover loss', name='Neuron Embedding', update = 'append')
            vis.line(X=np.array([coeff]), Y=np.array([loss_test[1]]), win='crossover acc', name='Neuron Embedding', update = 'append')

            log(f'Coefficient {coeff:.3f}: val loss: {loss_test[0]:.3f}, val acc: {loss_test[1]:.3f}', vis)

        vis.update_window_opts(
            win='crossover loss',
            opts=dict(
                showlegend = True,
                title = 'Crossentropy loss, neuron transplant',
                xlabel = 'Crossover coefficient',
                ylabel = 'Crossentropy loss',
                width = 600,
                height = 400
            )
        )

        vis.update_window_opts(
            win='crossover acc',
            opts=dict(
                showlegend = True,
                title = 'Accuracy, neuron transplant',
                xlabel= 'Crossover coefficient',
                ylabel= 'Accuracy',
                width= 600,
                height= 400,
                ytickmin=0,
                ytickmax=1
            ),
        )