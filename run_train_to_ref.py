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
from util import getdev
import loader
from models import scaled_dot, unscaled_dot, get_embeddings, get_layers, get_weights
from embedding import positional_encoding1d, positional_encoding2d

from torchsummary import summary

import embedding
import argparse

def train_to(train_model, ref_model, input_embedding = None):
    '''
    input_embedding: the input embedding to use. Layer will be locked (not trained)
    '''
    ref_weights = [
        dict(ref_model.named_parameters())['fc1.weight'].T,
        dict(ref_model.named_parameters())['fc2.weight'].T
    ]

    ref_biases = [
        dict(ref_model.named_parameters())['fc1.bias'],
        dict(ref_model.named_parameters())['fc2.bias']
    ]

    criterion = nn.MSELoss(reduction = 'mean')

    n_epochs = 1000
    log_stats = True

    log('Starting training', vis, win='training')
    start = time.time()
    loss_history = []
    optimizer = optim.Adam(train_model.parameters(), lr=0.01)
    
    # lock inputs to a specific encoding?
    if input_embedding is not None:
        sd = train_model.state_dict()
        sd['embs.0.weight']= input_embedding
        train_model.load_state_dict(sd)
        train_model.embs[0].weight.requires_grad = False
    
    for epoch in range(n_epochs):
        loss_train_running_total = 0
        count_train = 0

        # zero the parameter gradients
        optimizer.zero_grad()

        layers = get_layers(train_model.embs)
        weights = get_weights(layers)

        # If we're using weight biases and weight scaling

    #     for i in range(len(train_model.weight_biases)):
    #         weights[i] = weights[i] * train_model.weight_scaling[i] + train_model.weight_biases[i]

        biases = train_model.biases

        loss = 0
        loss += criterion(ref_weights[0], weights[0])
        loss += criterion(ref_weights[1], weights[1])
        loss += criterion(ref_biases[0], biases[0])/10
        loss += criterion(ref_biases[0], biases[0])/10

        loss.backward()
        optimizer.step()

        # print statistics
        if log_stats:
            loss_train = loss.item()
            log(f'Epoch {epoch+1:3d}, time {time.time() - start:.2f}- loss: {loss_train:.5f}', vis, win='training')
            loss_history.append(loss_train)
            vis.line(X=np.array([epoch]), Y=np.array([loss_train]), win='reference loss', name='train_loss', update = 'append')





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", "-r", type=str, help="path to source fully connected model", required=True)
    parser.add_argument("--save", "-s", type=str, help="location to save trained model", required=False)
    parser.add_argument("--dims", "-d", type=int, help="number of dimensions in embedding", default = 64)
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

    saved_model_path = args.ref
    if saved_model_path is not None:
        log(f'Loading model {saved_model_path}')
        ref_model_1.load_state_dict(torch.load(saved_model_path))


    test(ref_model_1, test_loader)

    emb_model_test = models.NeuronEmbedding(args.dims).to(dev)
    summary(emb_model_test, input_size = (28,28))
    train_to(emb_model_test, ref_model_1, input_embedding = None)

    if args.save:
        torch.save(emb_model_test.state_dict(), args.save)

    test(emb_model_test, test_loader)
