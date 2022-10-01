import torch
import torch.nn as nn
import numpy as np
import visdom
import matplotlib.pyplot as plt
import argparse
from torchinfo import summary

from src import config
from src.traintest import *
from src.util import getdev, count_parameters
from src import loader

import wandb

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type = str, help="configuration setting to use from settings.conf", required=True)
    parser.add_argument("--load", "-l", type=str, help="path to saved model to load", default=None)
    parser.add_argument("--dataset", "-d", type=str, help="dataset: MNIST, MNIST32", default='MNIST')
    parser.add_argument("--testonly", help="only run test, no training", action='store_true')

    # mostly used by wandb sweeps
    parser.add_argument("--conv_type", type=str, help="convolution type", default='')
    parser.add_argument("--batch_size", type=int, help="batch size", default=0)
    parser.add_argument("--learning_rate", type=float, help="learning rate", default=0)
    parser.add_argument("--max_lr", type=float, help="max learning rate", default=0.0)
    parser.add_argument("--weight_decay", type=float, help="weight decay", default=0.0)
 
    args = parser.parse_args()
    params = loader.read_config(config_name = args.config, custom_args = args)

    dev = getdev()
    model = loader.setup_model(params, dev = dev)

    # set up logging and visualization
    config.vis = visdom.Visdom()
    vis = config.vis
    log(params)
    wandb.init(project='neuron-embedding', config=params)
    
    train_loader, val_loader, test_loader, input_size = loader.load_data(dataset_name = params['dataset'], batch_size = params['batch_size'])

    log(f'Training size:{len(train_loader.dataset)}\nValidation size:{len(val_loader.dataset)}\nTesting size:{len(test_loader.dataset)}')

    summary(model, input_size = input_size, batch_dim = 0, device = dev)
    wandb.log({"Total parameters": count_parameters(model)}, step = 0)

    if args.load:
        log(f'Loading model {args.load}')
        model.load_state_dict(torch.load(args.load))
    
    loss_history = None
    if not args.testonly:
        # set up training
        log('Training Model')

        loss_history = train(model, dataloader = train_loader, params = params, val_loader=val_loader, log_stats = True, save_checkpoints = True)


    for test_loss in test(model, test_loader):
        wandb.log({"final_model_test_loss": test_loss[0], "final_model_test_acc": test_loss[1]})

    # also test the best model
    if loss_history is not None:
        best_acc_idx = np.argmax(loss_history[:, -1])
        best_step_num = int(loss_history[best_acc_idx, 0])
        model.load_state_dict(torch.load('./models'+f'/step{best_step_num}.pth'))
        log(f'Best model: step {best_step_num}', vis, win='training')
        wandb.log({"best_step_num": best_step_num})
        for test_loss in test(model, test_loader):
            wandb.log({"best_model_test_loss": test_loss[0], "best_model_test_acc": test_loss[1]})

