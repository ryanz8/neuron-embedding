import torch
import visdom
from torchinfo import summary

from src import config
from src import loader
from src.traintest import *
from src.util import getdev

import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--refpath", "-r", type=str, help="path to source fully connected model", required=True)
    parser.add_argument("--save", "-s", type=str, help="location to save trained model", required=False)
    parser.add_argument("--refconfig", "-c", type = str, help="configuration setting to use from settings.conf for direct weight model", required=True)
    parser.add_argument("--embconfig", "-e", type = str, help="configuration setting to use from settings.conf for embedding model", required=True)
    parser.add_argument("--dataset", "-d", type=str, help="dataset: MNIST, MNIST32", default='MNIST')
    args = parser.parse_args()

    dev = getdev()

    config.vis = visdom.Visdom()
    vis = config.vis
    
    ref_model, params = loader.setup_model(config_name = args.refconfig, dev = dev)
    train_loader, val_loader, test_loader, input_size = loader.load_data(dataset_name = params['dataset'], batch_size = params['batch_size'])

    ref_model.load_state_dict(torch.load(args.refpath))
    log('Ref initial evaluation', vis, win='training')
    test(ref_model, test_loader)

    emb_model, params = loader.setup_model(args.embconfig, dev)
    log('Emb initial evaluation', vis, win='training')
    summary(emb_model, input_size = input_size, batch_dim = 0, device = dev)
    test(emb_model, test_loader)

    train_to(emb_model = emb_model, ref_model = ref_model, params = params)

    if args.save:
        torch.save(emb_model.state_dict(), args.save)

    log('Emb final evaluation', vis, win='training')
    test(emb_model, test_loader)
