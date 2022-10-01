from .config import ROOT_DIR
from . import models
from .util import getdev
import torch
import torchvision
import torchvision.transforms as transforms
import configparser
import wandb

def load_data(dataset_name, batch_size):
    if dataset_name == 'MNIST':
        train_set, val_set, test_set, input_size = MNIST()
    elif dataset_name == 'MNIST_0_to_4':
        train_set, val_set, test_set, input_size = MNIST_subset(condition_0_to_4)
    elif dataset_name == 'MNIST_5_to_9':
        train_set, val_set, test_set, input_size = MNIST_subset(condition_5_to_9)
    elif dataset_name == 'MNIST32':
        train_set, val_set, test_set, input_size = MNIST32()
    elif dataset_name == 'CIFAR10':
        train_set, val_set, test_set, input_size = CIFAR10()
    elif dataset_name == 'CIFAR100':
        train_set, val_set, test_set, input_size = CIFAR100()
    else:
        raise(Exception("Unknown dataset name: " + dataset_name))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=True, num_workers=6, pin_memory=torch.cuda.is_available())
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=True, num_workers=6, pin_memory=torch.cuda.is_available())

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                            shuffle=False, num_workers=6, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader, input_size

def read_config(config_name, custom_args = None):
    '''
    Read the configuration file and build the param dict.

    args:
        config_name: string, configuration name (see settings.conf)
        custom_args: dict, will overwrite values specified in configuration file (e.g. for hyperparameter search)
    
    returns:
        params: the dict of parameters used, including any settings that were overwritten

    '''
    parser = configparser.ConfigParser()
    with open(ROOT_DIR + f'/configs/settings.conf') as f:
        parser.read_file(f)

    params = dict()

    param_list = [
        ('scheduler', 'str'),
        ('optimizer', 'str'),
        ('n_steps', 'int'),
        ('model_type', 'str'),
        ('dataset', 'str'),
        ('lr', 'float'),
        ('momentum', 'float'),
        ('weight_decay', 'float'),
        ('max_lr', 'float'),
        ('batch_size', 'int'),
        ('dropout_prob', 'float'),
        ('lr_ref', 'float'),
        ('n_steps_ref', 'int'),
        ('conv_lr_ref', 'float'),
        ('conv_n_steps_ref', 'int'),
        ('dims', 'int')
    ]
    for param_name, type in param_list:
        if param_name in parser[config_name]:
            config_name_to_use = config_name
        else:
            config_name_to_use = 'Default'
        if type == 'str':
            params[param_name] = parser[config_name_to_use][param_name]
        elif type == 'int':
            params[param_name] = int(parser[config_name_to_use][param_name])
        elif type == 'float':
            params[param_name] = float(parser[config_name_to_use][param_name])
        else:
            raise(Exception(f'Unknown type {type}'))

    if params['model_type'] == 'FC':
        params['layer_size'] = [int(x) for x in parser[config_name]['layer_size'].split(',')]

    elif params['model_type'] == 'Conv':
        params['conv_layer_size'] = [int(x) for x in parser[config_name]['conv_layer_size'].split(',')]
        params['lin_layer_size'] = [int(x) for x in parser[config_name]['lin_layer_size'].split(',')]
        params['kernel_size'] = int(parser[config_name]['kernel_size'])

    elif params['model_type'] == 'Emb':
        params['layer_size'] = [int(x) for x in parser[config_name]['layer_size'].split(',')]
        # params['dims'] = int(parser[config_name]['dims'])

    elif params['model_type'] == 'EmbConv':
        params['conv_layer_size'] = [int(x) for x in parser[config_name]['conv_layer_size'].split(',')]
        params['lin_layer_size'] = [int(x) for x in parser[config_name]['lin_layer_size'].split(',')]
        params['kernel_size'] = int(parser[config_name]['kernel_size'])
        params['dims'] = int(parser[config_name]['dims'])

    elif params['model_type'] == 'SeparableEmbConv':
        params['conv_layer_size'] = [int(x) for x in parser[config_name]['conv_layer_size'].split(',')]
        params['lin_layer_size'] = [int(x) for x in parser[config_name]['lin_layer_size'].split(',')]
        params['kernel_size'] = int(parser[config_name]['kernel_size'])
        params['dims'] = int(parser[config_name]['dims'])
        params['conv_type'] = parser[config_name]['conv_type']

    elif params['model_type'] == 'ResNet9':
        params['conv_layer_size'] = [int(x) for x in parser[config_name]['conv_layer_size'].split(',')]
        params['lin_layer_size'] = [int(x) for x in parser[config_name]['lin_layer_size'].split(',')]
        params['kernel_size'] = int(parser[config_name]['kernel_size'])
        params['dims'] = int(parser[config_name]['dims'])
        params['conv_type'] = parser[config_name]['conv_type']

    else:
        raise(Exception("Unknown model provided."))

    # overwrite these settings if specified on the command line
    if custom_args:
        if custom_args.batch_size:
            params['batch_size'] = int(custom_args.batch_size)
        if custom_args.learning_rate:
            params['learning_rate'] = float(custom_args.learning_rate)
        if custom_args.max_lr:
            params['max_lr'] = float(custom_args.max_lr)
        if custom_args.weight_decay:
            params['weight_decay'] = float(custom_args.weight_decay)
        if custom_args.conv_type:
            params['conv_type'] = custom_args.conv_type
    return params

def setup_model(params, dev = None):
    '''
    Sets up and returns PyTorch model using the configuration specified.

    args:
        config_name: string, configuration name (see settings.conf)
        dev: PyTorch device; if not specified will autodetect
        custom_args: dict, will overwrite values specified in configuration (e.g. for hyperparameter search)
    
    returns:
        model: initialized PyTorch model using the settings specified
        params: the dict of parameters used, including any settings that were overwritten
    '''
    if dev is None:
        dev = getdev()

    if params['model_type'] == 'FC':
        model = models.FCNet(
            layer_size = params['layer_size'],
            dropout_prob = params['dropout_prob']
        ).to(dev)
    elif params['model_type'] == 'Conv':
        model = models.ConvNet(
            conv_layer_size = params['conv_layer_size'],
            lin_layer_size = params['lin_layer_size'],
            kernel_size = params['kernel_size']
        ).to(dev)
    elif params['model_type'] == 'Emb':
        model = models.NeuronEmbedding(
            n_dims = params['dims'],
            layer_size = params['layer_size'],
            dropout_prob = params['dropout_prob']
        ).to(dev)
    elif params['model_type'] == 'EmbConv':
        model = models.NeuronEmbeddingConv(
            n_dims = params['dims'],
            conv_layer_size = params['conv_layer_size'],
            lin_layer_size = params['lin_layer_size'],
            kernel_size = params['kernel_size']
        ).to(dev)
    elif params['model_type'] == 'SeparableEmbConv':
        model = models.SeparableNeuronEmbeddingConv(
            n_dims = params['dims'],
            conv_layer_size = params['conv_layer_size'],
            lin_layer_size = params['lin_layer_size'],
            kernel_size = params['kernel_size'],
            conv_type = params['conv_type']
        ).to(dev)
    elif params['model_type'] == 'ResNet9':
        model = models.ResNet9(
            n_dims = params['dims'],
            conv_layer_size = params['conv_layer_size'],
            lin_layer_size = params['lin_layer_size'],
            kernel_size = params['kernel_size'],
            conv_type = params['conv_type']
        ).to(dev)
    else:
        raise(Exception("Unknown model provided."))

    return model

def CIFAR10():
    input_size = (3, 32, 32)

    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data = torchvision.datasets.CIFAR10(root=ROOT_DIR + '/data', train=True,
                                            download=True, transform=transform)
    train_set, val_set = torch.utils.data.random_split(data, [40000, 10000])

    test_set = torchvision.datasets.CIFAR10(root=ROOT_DIR + '/data', train=False,
                                        download=True, transform=transform)
    
    return train_set, val_set, test_set, input_size

def CIFAR100():
    input_size = (3, 32, 32)

    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data = torchvision.datasets.CIFAR10(root=ROOT_DIR + '/data', train=True,
                                            download=True, transform=transform)
    train_set, val_set = torch.utils.data.random_split(data, [40000, 10000])

    test_set = torchvision.datasets.CIFAR10(root=ROOT_DIR + '/data', train=False,
                                        download=True, transform=transform)
    
    return train_set, val_set, test_set, input_size

def _MNIST(transform):
    data = torchvision.datasets.MNIST(root=ROOT_DIR + '/data', train=True,
                                            download=True, transform=transform)
    train_set, val_set = torch.utils.data.random_split(data, [50000, 10000])

    test_set = torchvision.datasets.MNIST(root=ROOT_DIR + '/data', train=False,
                                        download=True, transform=transform)
    return train_set, val_set, test_set

def MNIST():
    input_size = (1, 28, 28)

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])

    return _MNIST(transform) + (input_size,)

def MNIST32():
    input_size = (1, 32, 32)

    transform = transforms.Compose(
    [torchvision.transforms.Resize(32), transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])

    return _MNIST(transform) + (input_size,)

def condition_0_to_4(targets):
    return torch.where(torch.logical_and(targets >= 0, targets <= 4))[0]

def condition_5_to_9(targets):
    return torch.where(torch.logical_and(targets >= 5, targets <= 9))[0]

def MNIST_subset(condition):
    input_size = (1, 28, 28)

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])

    train_data = torchvision.datasets.MNIST(root=ROOT_DIR + '/data', train=True,
                                            download=True, transform=transform)

    inds_train_subset = condition(train_data.targets)
    # hack since PyTorch doesn't like the targets starting at 5
    train_data.targets[inds_train_subset] -= min(train_data.targets[inds_train_subset])

    train_data_subset = torch.utils.data.Subset(train_data, inds_train_subset)

    train_set, val_set = torch.utils.data.random_split(train_data_subset, [25000, len(train_data_subset) - 25000])

    test_data = torchvision.datasets.MNIST(root=ROOT_DIR + '/data', train=False,
                                        download=True, transform=transform)

    inds_test_subset = condition(test_data.targets)
    test_data.targets[inds_test_subset] -= min(test_data.targets[inds_test_subset])
    test_set = torch.utils.data.Subset(test_data, inds_test_subset)

    return train_set, val_set, test_set, input_size
