import torch

def interpolate(model1, model2, coefficient, layer_names = None):
    '''
    Does linear interpolation between the weights of two models.

    model1: model to interpolate from (coefficient 0)
    model2: model to interpolate to (coefficient 1)
    coefficient: coefficient of interpolation: 0 = model 1, 1 = model 2

    returns: state dict for the interpolated model.
    '''

    # seems like it's safer to do it via the state_dict?
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    # only interpolate specific layers if asked for, copy others from model 1
    if layer_names:
        interp_dict = {k: torch.lerp(sd1[k], sd2[k], coefficient if k in layer_names else 0) for k in sd1.keys()}
    else:
        interp_dict = {k: torch.lerp(sd1[k], sd2[k], coefficient) for k in sd1.keys()}
    return interp_dict


def swap_neurons(model1, model2, layer_names, neuron_ids):
    '''
    Swaps neurons specified by neuron_ids in layer_name from model2 to model1.
    
    model1: destination model for neurons
    model2: source model for neurons
    layer_names: list of layers to swap
    n_neurons: list of lists of neurons to swap in each layer, in the same order as layer_names

    returns: state dict for the model containing the new neurons.
    '''
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()
    
    sd_new = sd1.copy()
    
    for layer_name, ids in zip(layer_names, neuron_ids):
        # make sure we're not modifying the existing tensor in the endpoint models
        sd_new[layer_name] = sd1[layer_name].clone().detach()
        sd_new[layer_name][ids] = sd2[layer_name][ids].clone().detach()
    return sd_new

def transfer_layers(model_to, model_from, layer_names):
    '''
    Transfers entire layers from model_from to model_to.
    
    model_to: destination model to transfer layers to
    model_from: source model to transfer layers from

    layer_names: list of layers to transfer
    
    returns: state dict for the model containing the new layers.
    '''
    sd_to = model_to.state_dict().copy()
    sd_from = model_from.state_dict().copy()
    for layer_name in layer_names:
        sd_to[layer_name] = sd_from[layer_name].clone().detach()
    
    return sd_to

def swap_outgoing(model1, model2, layer_names, neuron_ids):
    '''
    Similar to swap_neurons, but swaps the outgoing connections from a neuron rather than incoming.
    This is intended for the direct weight representation.
    
    model1: destination model for neurons
    model2: source model for neurons
    layer_names: list of layers to swap. Since the outgoing weights reside in the NEXT layer,
                 this must be the name of the next layer.
    n_neurons: list of lists of neurons to swap in each layer, in the same order as layer_names

    returns: state dict for the model containing the new neurons.
    '''
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()
    
    sd_new = sd1.copy()
    
    for layer_name, ids in zip(layer_names, neuron_ids):
        # make sure we're not modifying the existing tensor in the endpoint models
        sd_new[layer_name] = sd1[layer_name].clone().detach()
        sd_new[layer_name].T[ids] = sd2[layer_name].T[ids].clone().detach()
#         print('shapes', sd_new[layer_name].T[ids].shape, sd2[layer_name].T[ids].shape)
    return sd_new
