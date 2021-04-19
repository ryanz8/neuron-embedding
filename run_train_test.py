import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np

import visdom
import matplotlib.pyplot as plt

import models
import config
from traintest import *
from util import getdev
import loader
import argparse

from torchsummary import summary

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, help="model name: FC, Conv, Emb, EmbConv", required=True)
    parser.add_argument("--load", "-l", type=str, help="path to saved model to load", default=None)
    parser.add_argument("--dims", "-d", type=int, help="number of dimensions in embedding", default=64)
    parser.add_argument("--epochs", "-e", type=int, help="number of epochs to train", default=100)
    args = parser.parse_args()

    dev = getdev()

    config.vis = visdom.Visdom()
    vis = config.vis
    
    # only this dataset is supported for now
    dataset = 'MNIST'
    if dataset == 'MNIST':
        train_loader, val_loader, test_loader = loader.MNIST()
        input_size = (1, 28, 28)

    log(f'Training size:{len(train_loader.dataset)}\nValidation size:{len(val_loader.dataset)}\nTesting size:{len(test_loader.dataset)}')

    if args.model == 'FC':
        model = models.FCNet().to(dev)
    elif args.model == 'Conv':
        model = models.ConvNet().to(dev)
    elif args.model == 'Emb':
        model = models.NeuronEmbedding(args.dims).to(dev)
    elif args.model == 'EmbConv':
        model = models.NeuronEmbeddingConv(args.dims).to(dev)
    else:
        raise(Exception("Unknown model provided."))
    
    summary(model, input_size=input_size, device = dev)

    loss_history = None
    if args.load:
        log(f'Loading model {args.load}')
        model.load_state_dict(torch.load(args.load))
    else:
        # set up training
        log('Training Model')
        training_epochs = args.epochs
        lr = 0.002
        criterion = nn.CrossEntropyLoss(reduction = 'sum')
        optimizer = optim.Adam

        loss_history = train(model,train_loader, training_epochs,
            lr = lr, criterion = criterion, opt = optimizer, val_loader=val_loader, log_stats = True, save_checkpoints = True)


    test(model, test_loader)

    # also test the best model
    if loss_history is not None:
        best = np.argmin(loss_history)
        model.load_state_dict(torch.load('./models'+f'/epoch{best+1}.pth'))
        log(f'Best model: epoch {best}')
        test(model, test_loader)
