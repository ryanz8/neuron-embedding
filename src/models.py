import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from . import embedding

import wandb


def get_embeddings(emb):
    '''
    Helper function to get all the embeddings for a layer.
    Technically we could just grab the weight tensor directly, but going through the embedding layer enables us to use some options
    '''
    return emb(torch.LongTensor([i for i in range(emb.num_embeddings)]).to(emb.weight.device))

def scaled_dot(layer1, layer2):
    '''
    Scaled dot product, like the one used in Transformer
    '''
    return torch.matmul(layer1, layer2.transpose(0, 1))/(layer1.shape[-1]) # **0.5

def unscaled_dot(layer1, layer2):
    '''
    Unscaled (vanilla) dot product
    '''
    return torch.matmul(layer1, layer2.transpose(0, 1))

def sinh_dot(layer1, layer2):
    '''
    Unscaled (vanilla) dot product
    '''
    return torch.sinh(torch.matmul(layer1, layer2.transpose(0, 1)))

def get_layers(embs):
    '''
    Gets the embeddings for all neurons in a layer.
    
    Input: list of embedding modules
    Output: list of tensors (embeddings)
    '''
    return [get_embeddings(emb) for emb in embs]

def get_weights(embs, kernel = unscaled_dot, env = None):
    '''
    Calculates the weight matrices by calculating dot product alignment between consecutive pairs of layers.
    I.e., if the input is [layer1, layer2, layer3] we calculate [dot(layer1, layer2), dot(layer2, layer3)]
    
    Input: layers: list of layers, where each layer is represented as the tensor of neuron embeddings
           env:    optional environment function (probably a linear feedforward net or similar)
    Output: list of weight tensors
    '''
    layers = get_layers(embs)
    if env is None:
        return [kernel(l1, l2) for l1,l2 in zip(layers[:-1], layers[1:])]
    else:
        return [kernel(env(l1), env(l2)) for l1,l2 in zip(layers[:-1], layers[1:])]

def get_conv_weights(conv_embs, field_embs, kernel_size):
    '''
    Builds the weight tensor for the convolution kernel (4d).
    
    If there are n neurons in the current layer and m neurons in the previous layer,
    takes a kernel/receptive field embedding with dimension (n*k*k, d) and a layer
    embedding with dimension (m, d) for the previous layer and (n, d) for the current layer.

    Adds a field embedding to each of the n neuron embeddings in the current layer and then
    calculates weights wrt to the previous layer. Note that this means one field embedding per one
    neuron in the current layer, broadcast across the m neurons in the previous layer. Weights will
    not be the same since we still calculate dot product wrt to each of the m neurons, but may be
    correlated (not necessarily a problem since this is also the case in regular nets).

    Input: conv_embs: list of Embedding layers for the main neuron embedding
           field_embs: list of Embedding layers for the receptive fields (1 per neuron)
           kernel_size: size of the kernel to reshape the field embeddings to the proper size
    Output: list of weight tensors
    '''
    conv_layers = get_layers(conv_embs)
    conv_layers_reshaped = [t.reshape((-1, 1, 1, t.shape[-1])) for t in conv_layers]
    conv_fields = get_layers(field_embs)
    conv_fields_reshaped = [t.reshape((-1, kernel_size, kernel_size, t.shape[-1])) for t in conv_fields]

    # build compound embedding
    conv_output_embs = [layer + field for layer,field in zip(conv_layers_reshaped[1:], conv_fields_reshaped[:])]

    # weight calculation
    conv_weights = [ (output_emb @ input_emb.T).permute(0, 3, 1, 2).contiguous() for output_emb, input_emb in zip(conv_output_embs, conv_layers[:-1]) ]
    return conv_weights

def init_bias(param, fan_in):
    '''
    Initialize the biases, since we had to define the manually.
    Similar to bias initialization in PyTorch linear layer.
    
    Input:  param: the bias to be initialized
            fan_in: number of incoming connections (usually the size of the previous layer)
    '''
    bound = 1 / ((fan_in)**0.5)
    nn.init.uniform_(param, -bound, bound)

@torch.no_grad()
def reset_param(m):
    '''
    Reset parameters for a single module. Used to efficiently initialize the model with apply()

    Input: m: the module to be reset.
    '''
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

###########################################################################################

class FCNet(nn.Module):
    '''
    A simple fully connected model.
    '''
    def __init__(self, layer_size, dropout_prob):
        super(FCNet, self).__init__()
        self.layer_size = layer_size

        self.linears = nn.ModuleList([nn.Linear(n_in, n_out) for n_in, n_out in zip(self.layer_size[:-1], self.layer_size[1::])])
        self.dropout = nn.Dropout(p = dropout_prob)
        # self.fc1 = nn.Linear(self.layer_size[0], self.layer_size[1])
        # self.fc2 = nn.Linear(self.layer_size[1], self.layer_size[2])

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for l in self.linears[:-1]:
            x = F.relu(l(x))
            x = self.dropout(x)
        
        # no relu on last layer
        out = self.linears[-1](x)
        # out = F.relu(self.fc1())
        return out

# What if we also add the neuron embeddings to the input?

class NeuronEmbedding(nn.Module):
    '''
    Fully connected model using neuron embeddings.
    '''
    def __init__(self, n_dims, layer_size, dropout_prob):
        super().__init__()
        self.n_dims = n_dims
        self.layer_size = layer_size

        self.biases = nn.ParameterList([
            nn.Parameter(torch.Tensor(ls)) for ls in self.layer_size[1:]
        ])
        
        self.embs = nn.ModuleList([
            nn.Embedding(num_embeddings = ls, embedding_dim = self.n_dims) #, max_norm=self.n_dims**0.5)
            for ls in self.layer_size
        ])
        # self.env = nn.Linear(self.n_dims, self.dot_product_dim)
        self.dropout = nn.Dropout(p = dropout_prob)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for i, param in enumerate(self.biases):
            init_bias(param, self.layer_size[i])

        for i, emb in enumerate(self.embs):
            # this rescaled initialization SHOULD remove the need for scaled_dot
            nn.init.normal_(emb.weight, std=1.0/(self.n_dims**0.5))

    def forward(self, x, return_intermediate = False):
        weights = get_weights(self.embs)

        out = x.view(x.shape[0], -1)
        outlist = []

        # iterate through layers
        for W, b in zip(weights[:-1], self.biases[:-1]):
            out = F.relu(out @ W + b)
            out = self.dropout(out)

            if return_intermediate:
                outlist.append(out.detach().clone())

        # last one should not have a ReLU
        out = out @ weights[-1] + self.biases[-1]

        if return_intermediate:
            return out, outlist

        return out

class NeuronEmbeddingWeightBiases(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_dims = 16
        self.layer_size = [28*28, 400, 10]

        self.biases = nn.ParameterList([
            nn.Parameter(torch.Tensor(ls)) for ls in self.layer_size[1:]
        ])
        self.weight_biases = nn.ParameterList([
            nn.Parameter(torch.Tensor(ls)) for ls in self.layer_size[1:]
        ])
        self.weight_scaling = nn.ParameterList([
            nn.Parameter(torch.Tensor(ls)) for ls in self.layer_size[1:]
        ])
        
        self.embs = nn.ModuleList([
            nn.Embedding(num_embeddings = ls, embedding_dim = self.n_dims) #, max_norm=self.n_dims**0.5)
            for ls in self.layer_size
        ])
        # self.env = nn.Linear(self.n_dims, self.dot_product_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for i, param in enumerate(self.biases):
            init_bias(param, self.layer_size[i])

        for i, param in enumerate(self.weight_biases):
            init_bias(param, self.layer_size[i])

        for i, param in enumerate(self.weight_scaling):
            # I'm not really sure what makes sense, just going to try this for now
            # do negative values make sense?
            # maybe? it would penalize ones that are close
            nn.init.normal_(param, mean=1.0)

        for i, emb in enumerate(self.embs):
            # this rescaled initialization SHOULD remove the need for scaled_dot
            nn.init.normal_(emb.weight, std=1.0/(self.n_dims**0.5))

    def forward(self, x, return_intermediate = False):
        weights = get_weights(self.embs)

        for i in range(len(self.weight_biases)):
            weights[i] = weights[i] * self.weight_scaling[i] + self.weight_biases[i]
        
        out0 = x.view(x.shape[0], -1)
        out1 = F.relu(out0 @ weights[0] + self.biases[0])
        out2 = out1 @ weights[1] + self.biases[1]
        if return_intermediate:
            return out1, out2
        return out2


class ConvNet(nn.Module):
    '''
    A simple convolutional network.
    '''
    def __init__(self, conv_layer_size = [1, 10, 40], lin_layer_size = [1000, 10], kernel_size = 3) :
        super().__init__()
        self.conv_layer_size = conv_layer_size
        self.lin_layer_size = lin_layer_size
        self.kernel_size = kernel_size
        

        # self.conv1 = nn.Conv2d(self.conv_layer_size[0], self.conv_layer_size[1], (3, 3))
        # self.conv2 = nn.Conv2d(self.conv_layer_size[1], self.conv_layer_size[2], (3, 3))
        self.maxpool = nn.MaxPool2d((2,2))
        # self.lin1 = nn.Linear(self.lin_layer_size[0], 10)

        self.convs = nn.ModuleList([nn.Conv2d(n_in, n_out, kernel_size=self.kernel_size, padding='same') for n_in, n_out in zip(self.conv_layer_size[:-1], self.conv_layer_size[1::])])
        self.batchnorms = nn.ModuleList([nn.BatchNorm2d(n_out, affine = False) for n_out in self.conv_layer_size[1::]])
        self.linears = nn.ModuleList([nn.Linear(n_in, n_out) for n_in, n_out in zip(self.lin_layer_size[:-1], self.lin_layer_size[1::])])

    def forward(self, x, return_intermediate = False):

        outlist = []

        for conv, batchnorm in zip(self.convs, self.batchnorms):
            x = F.relu(conv(x))
            x = batchnorm(x)
            x = self.maxpool(x)
            if return_intermediate:
                outlist.append(x.detach().clone())

        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.maxpool(x)

        x = torch.flatten(x, 1) # flatten out channel, w, h into one dimension
#        x = x.view(x.shape[0], -1) 

        for lin in self.linears[:-1]:
            x = F.relu(lin(x))
            if return_intermediate:
                outlist.append(x.detach().clone())

        out = self.linears[-1](x)
        return out


class NeuronEmbeddingConv(nn.Module):
    '''
    Reconstructs the full convolutional kernel using neuron embeddings.
    See get_conv_weights for explanation.

    conv_layer_size: list of number of neurons/features/channels in each convolutional layer. First entry is the number of input channels
    lin_layer_size: list of number of neurons/features in the linear layers. First entry is the number of features after image is flattened
    '''
    def __init__(self, n_dims = 32, conv_layer_size = [1, 10, 40], lin_layer_size = [1000, 10], kernel_size = 3):
        super().__init__()
        self.n_dims = n_dims
        self.kernel_size = kernel_size
        self.conv_layer_size = conv_layer_size
        self.lin_layer_size = lin_layer_size

        self.maxpool = nn.MaxPool2d((2,2))

        self.conv_biases = nn.ParameterList([
            nn.Parameter(torch.Tensor(ls)) for ls in self.conv_layer_size[1:]
        ])
        self.lin_biases = nn.ParameterList([
            nn.Parameter(torch.Tensor(ls)) for ls in self.lin_layer_size[1:]
        ])
        
        # we have a base receptive field shared between all the channels in a neuron
        # this is possibly a dumb way to do it, but we'll never know unless we try
        # it's sort of like a "factorized" embedding
        self.field_embs = nn.ModuleList([
            nn.Embedding(num_embeddings=ls*self.kernel_size**2, embedding_dim=self.n_dims)
            for ls in self.conv_layer_size[1:]
        ])

        self.conv_embs = nn.ModuleList([
            nn.Embedding(num_embeddings = ls, embedding_dim = self.n_dims) #, max_norm=self.n_dims**0.5)
            for ls in self.conv_layer_size
        ])
        self.lin_embs = nn.ModuleList([
            nn.Embedding(num_embeddings = ls, embedding_dim = self.n_dims) #, max_norm=self.n_dims**0.5)
            for ls in self.lin_layer_size
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for i, param in enumerate(self.conv_biases):
            init_bias(param, self.conv_layer_size[i])

        for i, param in enumerate(self.lin_biases):
            init_bias(param, self.lin_layer_size[i])

        for i, emb in enumerate(chain(self.field_embs, self.conv_embs, self.lin_embs)):
            # this rescaled initialization SHOULD remove the need for scaled_dot
            nn.init.normal_(emb.weight, std=1.0/(self.n_dims**0.5))

    def forward(self, x, return_intermediate = False):
        conv_weights = get_conv_weights(self.conv_embs, self.field_embs, self.kernel_size)
        lin_weights = get_weights(self.lin_embs)

        # actual forward step
        out = x
        outlist = []

        for W, b, in zip(conv_weights, self.conv_biases):
            out = F.relu(F.conv2d(out, weight=W, bias=b, padding='same'))
            out = self.maxpool(out)
            if return_intermediate:
                outlist.append(out.detach().clone())

        out = out.reshape(out.shape[0], -1) # flatten out channel, w, h into one dimension

        # iterate through layers
        for W, b in zip(lin_weights[:-1], self.lin_biases[:-1]):
            out = F.relu(out @ W + b)
            if return_intermediate:
                outlist.append(out.detach().clone())

        # last one should not have a ReLU
        out = out @ lin_weights[-1] + self.lin_biases[-1]

        if return_intermediate:
            return out, outlist
        return out

# class SeparableNeuronEmbeddingConv(nn.Module):
#     '''
#     Reverse depthwise separable convolution (Blueprint Separable Convolution-U)
#     using neuron embeddings.
#     '''
#     def __init__(self, n_dims = 32, conv_layer_size = [1, 10, 40], lin_layer_size = [1000, 10], kernel_size = 3):
#         super().__init__()
#         self.n_dims = n_dims
#         self.kernel_size = kernel_size
#         self.conv_layer_size = conv_layer_size
#         self.lin_layer_size = lin_layer_size

#         self.maxpool = nn.MaxPool2d((2,2))
        
#         self.pointwise_embs = nn.ModuleList([
#             nn.Embedding(num_embeddings = ls, embedding_dim = self.n_dims) #, max_norm=self.n_dims**0.5)
#             for ls in self.conv_layer_size
#         ])
#         self.depthwise_convs = nn.ModuleList([
#             nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
#                            groups=out_channels, bias=True, padding='same')
#             for out_channels in self.conv_layer_size[1:]
#         ])

#         self.batchnorms = nn.ModuleList([nn.BatchNorm2d(n_out, affine = False) for n_out in self.conv_layer_size[1::]])

#         self.lin_embs = nn.ModuleList([
#             nn.Embedding(num_embeddings = ls, embedding_dim = self.n_dims) #, max_norm=self.n_dims**0.5)
#             for ls in self.lin_layer_size
#         ])
#         self.lin_biases = nn.ParameterList([
#             nn.Parameter(torch.Tensor(ls)) for ls in self.lin_layer_size[1:]
#         ])

#         self.apply(reset_param)

#     def reset_parameters(self):
#         with torch.no_grad():
#             for layer in self.children():
#                 layer.apply(reset_param)
            
#             for i, param in enumerate(self.lin_biases):
#                 init_bias(param, self.lin_layer_size[i])

#             for i, emb in enumerate(chain(self.pointwise_embs, self.lin_embs)):
#                 # this rescaled initialization SHOULD remove the need for scaled_dot
#                 nn.init.normal_(emb.weight, std=1.0/(self.n_dims**0.5))

#     def forward(self, x, return_intermediate = False):
#         lin_weights = get_weights(self.lin_embs)
#         pointwise_weights = get_weights(self.pointwise_embs)
        
#         # actual forward step
#         out = x
#         outlist = []
        
#         # conv layers
#         for layer_num, (pointwise_weight, depthwise_conv, batchnorm) in enumerate(zip(pointwise_weights, self.depthwise_convs, self.batchnorms)):
#             # reverse order depthwise separable: do pointwise first, then depthwise
#             out = F.conv2d(out, weight=pointwise_weight.T.unsqueeze(-1).unsqueeze(-1), padding='same')
#             out = F.relu(depthwise_conv(out))
#             out = batchnorm(out)
#             out = self.maxpool(out)
#             if return_intermediate:
#                 outlist.append(out.detach().clone())

#         out = out.reshape(out.shape[0], -1) # flatten out channel, w, h into one dimension

#         # linear layers
#         for W, b in zip(lin_weights[:-1], self.lin_biases[:-1]):
#             out = F.relu(out @ W + b)
#             if return_intermediate:
#                 outlist.append(out.detach().clone())

#         # last one should not have a ReLU
#         out = out @ lin_weights[-1] + self.lin_biases[-1]

#         if return_intermediate:
#             return out, outlist
#         return out

class SeparableNeuronEmbeddingConv(nn.Module):
    '''
    Reverse depthwise separable convolution (Blueprint Separable Convolution-U)
    using neuron embeddings.

    Args:
        n_dims: Number of dimensions in the embedding. Unused for non-embedding models
        conv_layer_size: List of layer widths of the convolutional layers,
            starting with the number of features in the input layer.
        lin_layer_size: List of layer widths of the feedforward layers,
            starting with the number of features coming out of the last convolutional layer.
        kernel_size: The kernel size of the convolution.
        conv_type: Type of convolution to use. Options are:
            'standard': Standard direct weight representation of full convolutions
            'separable': Reverse order depthwise separable convolutions, direct weight representation
            'embedded': Embedded representation, reverse order depthwise separable convolutions.
                Linear feedforward layers will also be represented with embeddings.

    '''
    def __init__(self, n_dims = 32, conv_layer_size = [1, 10, 40], lin_layer_size = [1000, 10], kernel_size = 3, conv_type = 'standard'):
        super().__init__()
        self.n_dims = n_dims
        self.kernel_size = kernel_size
        self.conv_layer_size = conv_layer_size
        self.lin_layer_size = lin_layer_size

        self.maxpool = nn.MaxPool2d((2,2))
        self.batchnorms = nn.ModuleList([nn.BatchNorm2d(n_out, affine = False, track_running_stats=False) for n_out in self.conv_layer_size[1::]])
        
        self.conv_type = conv_type
        
        if self.conv_type == 'standard':
            self.convs = nn.ModuleList([nn.Conv2d(n_in, n_out, kernel_size=self.kernel_size, padding='same') for n_in, n_out in zip(self.conv_layer_size[:-1], self.conv_layer_size[1::])])
        elif self.conv_type == 'separable':
            self.pointwise_convs = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False, padding = 'same')
                for in_channels, out_channels in zip(self.conv_layer_size[:-1], self.conv_layer_size[1::])
            ])

            self.depthwise_convs = nn.ModuleList([
                nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, 
                               groups=out_channels, bias=True, padding='same')
                for out_channels in self.conv_layer_size[1:]
            ])
        elif self.conv_type == 'embedded':
            self.pointwise_embs = nn.ModuleList([
                nn.Embedding(num_embeddings = ls, embedding_dim = self.n_dims) #, max_norm=self.n_dims**0.5)
                for ls in self.conv_layer_size
            ])

            self.depthwise_convs = nn.ModuleList([
                nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, 
                               groups=out_channels, bias=True, padding='same')
                for out_channels in self.conv_layer_size[1:]
            ])

        # do embedding for linear too if we did for the convs
        if self.conv_type == 'embedded':
            self.lin_embs = nn.ModuleList([
                nn.Embedding(num_embeddings = ls, embedding_dim = self.n_dims) #, max_norm=self.n_dims**0.5)
                for ls in self.lin_layer_size
            ])
            self.lin_biases = nn.ParameterList([
                nn.Parameter(torch.Tensor(ls)) for ls in self.lin_layer_size[1:]
            ])
        else:
            self.linears = nn.ModuleList([nn.Linear(n_in, n_out) for n_in, n_out in zip(self.lin_layer_size[:-1], self.lin_layer_size[1::])])


        self.apply(reset_param)

    def reset_parameters(self):
        with torch.no_grad():
            for layer in self.children():
                layer.apply(reset_param)
            
            # custom initialization for these, since PyTorch doesn't know what they'll be used for
            if hasattr(self, 'lin_biases'):
                for i, param in enumerate(self.lin_biases):
                    init_bias(param, self.lin_layer_size[i])

            if hasattr(self, 'pointwise_embs'):
                for i, emb in enumerate(chain(self.pointwise_embs, self.lin_embs)):
                    # this rescaled initialization SHOULD remove the need for scaled_dot
                    # nn.init.normal_(emb.weight, std=1.0/(self.n_dims**0.5))
                    nn.init.uniform_(emb.weight, a = -(3.0/self.n_dims)**0.5, b = (3.0/self.n_dims)**0.5)
    def forward(self, x, return_intermediate = False):
        
        def do_conv(x, layer_num):
            if self.conv_type == 'standard':
                return self.convs[layer_num](x)
            elif self.conv_type == 'separable':
                return self.depthwise_convs[layer_num](self.pointwise_convs[layer_num](x))
            elif self.conv_type == 'embedded':
                return self.depthwise_convs[layer_num](
                    F.conv2d(x, weight=pointwise_weights[layer_num].T.unsqueeze(-1).unsqueeze(-1))
                )
            
        
        if self.conv_type == 'embedded':
            lin_weights = get_weights(self.lin_embs)
            pointwise_weights = get_weights(self.pointwise_embs)
        
        # actual forward step
        out = x
        outlist = []
        
        # conv layers
        for layer_num in range(len(self.conv_layer_size)-1):
            out = F.relu(do_conv(out, layer_num))
            if return_intermediate:
                outlist.append(out.detach().clone())
            out = self.batchnorms[layer_num](out)
            out = self.maxpool(out)
            if return_intermediate:
                outlist.append(out.detach().clone())

        out = out.reshape(out.shape[0], -1) # flatten out channel, w, h into one dimension

        if self.conv_type == 'embedded':
            # linear layers
            for W, b in zip(lin_weights[:-1], self.lin_biases[:-1]):
                out = F.relu(out @ W + b)
                if return_intermediate:
                    outlist.append(out.detach().clone())
            # last one should not have a ReLU
            out = out @ lin_weights[-1] + self.lin_biases[-1]
        else:
            for lin in self.linears[:-1]:
                out = F.relu(lin(out))
                if return_intermediate:
                    outlist.append(out.detach().clone())
            # last one should not have a ReLU
            out = self.linears[-1](out)

        if return_intermediate:
            return out, outlist
        return out

class ResNet9(SeparableNeuronEmbeddingConv):
    '''
    Specific 9-layer ResNet architecture from https://davidpicard.github.io/pdf/lucky_seed.pdf, based on
    the architecture developed in https://myrtle.ai/learn/how-to-train-your-resnet/.
    '''
    def __init__(self, n_dims = 32, conv_layer_size = [1, 10, 40], lin_layer_size = [1000, 10], kernel_size = 3, conv_type = 'standard'):
        if len(conv_layer_size) != 9 or len(lin_layer_size) != 2:
            raise(Exception("len(conv_layer_size) must be 9 and len(lin_layer_size) must be 2."))
        super().__init__(n_dims, conv_layer_size, lin_layer_size, kernel_size, conv_type)
    
    def forward(self, x, return_intermediate = False):
        
        def do_conv(x, layer_num):
            if self.conv_type == 'standard':
                return self.convs[layer_num](x)
            elif self.conv_type == 'separable':
                return self.depthwise_convs[layer_num](self.pointwise_convs[layer_num](x))
            elif self.conv_type == 'embedded':
                return self.depthwise_convs[layer_num](
                    F.conv2d(x, weight=pointwise_weights[layer_num].T.unsqueeze(-1).unsqueeze(-1))
                )
        
        def conv_block(x, layer_num):
            batchnorm = self.batchnorms[layer_num]
            out = F.relu(batchnorm(do_conv(x, layer_num)))
            return out
        
        
        if self.conv_type == 'embedded':
            lin_weights = get_weights(self.lin_embs)
            pointwise_weights = get_weights(self.pointwise_embs)
        
        # actual forward step
        outlist = []

        x = conv_block(x, 0)
        x = conv_block(x, 1)
        x = self.maxpool(x)

        x_res = conv_block(x, 2)
        x_res = conv_block(x_res, 3)
        x = x + x_res

        x = conv_block(x, 4)
        x = self.maxpool(x)
        x = conv_block(x, 5)
        x = self.maxpool(x)

        x_res = conv_block(x, 6)
        x_res = conv_block(x_res, 7)
        x = x + x_res
        
        x = torch.mean(x, (-1, -2)) # global average pooling
#        x = torch.amax(x, (-1, -2)) # global max pooling

        out = x.reshape(x.shape[0], -1) # flatten out channel, w, h into one dimension

        if self.conv_type == 'embedded':
            # linear layers
            for W, b in zip(lin_weights[:-1], self.lin_biases[:-1]):
                out = F.relu(out @ W + b)
                if return_intermediate:
                    outlist.append(out.detach().clone())
            # last one should not have a ReLU
            out = out @ lin_weights[-1] + self.lin_biases[-1]
        else:
            for lin in self.linears[:-1]:
                out = F.relu(lin(out))
                if return_intermediate:
                    outlist.append(out.detach().clone())
            # last one should not have a ReLU
            out = self.linears[-1](out)
            # out = out * 0.125 # I'm not sure why they did this in the paper, temporary

        if return_intermediate:
            return out, outlist
        return out

class NeuronEmbedding2Sided(nn.Module):
    '''
    (Outdated) Two-sided embeddings, with separate input and output embeddings.
    '''
    def __init__(self):
        super(NeuronEmbedding2Sided, self).__init__()
        self.n_dims = 16

        self.emb1 = nn.Embedding(num_embeddings = 28*28, embedding_dim = self.n_dims, max_norm=self.n_dims**0.5)
        self.emb2_1 = nn.Embedding(num_embeddings = 400, embedding_dim = self.n_dims, max_norm=self.n_dims**0.5)
        self.emb2_2 = nn.Embedding(num_embeddings = 400, embedding_dim = self.n_dims, max_norm=self.n_dims**0.5)
        self.emb3 = nn.Embedding(num_embeddings = 10, embedding_dim = self.n_dims, max_norm=self.n_dims**0.5)
        self.dev_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        layer1 = get_embeddings(self.emb1)
        layer2_1 = get_embeddings(self.emb2_1)
        layer2_2 = get_embeddings(self.emb2_2)
        layer3 = get_embeddings(self.emb3)

        weight1 = scaled_dot(layer1, layer2_1)
        weight2 = scaled_dot(layer2_2, layer3)

        out = x.view(x.shape[0], -1)
        out = F.relu(torch.matmul(out, weight1))
        out = (torch.matmul(out, weight2))
        return out

class DeepNeuronEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_dims = 16
        self.layer_size = [28*28, 600, 600, 400, 200, 10]

        self.biases = nn.ParameterList([
            nn.Parameter(torch.Tensor(ls)) for ls in self.layer_size[1:]
        ])
        self.reset_parameters()
        
        self.embs = nn.ModuleList([
            nn.Embedding(num_embeddings = ls, embedding_dim = self.n_dims) #, max_norm=self.n_dims**0.5)
            for ls in self.layer_size
        ])
        # self.env = nn.Linear(self.n_dims, self.dot_product_dim)

    def reset_parameters(self):
        for i, param in enumerate(self.biases):
            # these bounds are based on fan_in, so it's the size of the previous layer
            bound = 1 / ((self.layer_size[i])**0.5)
            nn.init.uniform_(param, -bound, bound)

        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, return_intermediate = False):
        weights = get_weights(self.embs)
        
        out = x.view(x.shape[0], -1)
        out = F.relu(out @ weights[0] + self.biases[0])
        out = F.relu(out @ weights[1] + self.biases[1])
        out = F.relu(out @ weights[2] + self.biases[2])
        out = F.relu(out @ weights[3] + self.biases[3])
        out = out @ weights[4] + self.biases[4]
        if return_intermediate:
            return out
        return out
    
class PositionalNeuronEmbedding(nn.Module):
    def __init__(self):
        super(PositionalNeuronEmbedding, self).__init__()
        self.n_dims = 16
        self.layer_size = [28*28, 400, 10]
        # self.dot_product_dim = 8

        self.bias2 = nn.Parameter(torch.Tensor(self.layer_size[1]))
        self.bias3 = nn.Parameter(torch.Tensor(self.layer_size[2]))
        self.reset_parameters()
        
        self.emb1 = nn.Embedding(num_embeddings = self.layer_size[0], embedding_dim = self.n_dims)#, max_norm=self.n_dims**0.5)
        self.emb2 = nn.Embedding(num_embeddings = self.layer_size[1], embedding_dim = self.n_dims)#, max_norm=self.n_dims**0.5)
        self.emb3 = nn.Embedding(num_embeddings = self.layer_size[2], embedding_dim = self.n_dims)#, max_norm=self.n_dims**0.5)

        # self.env = nn.Linear(self.n_dims, self.dot_product_dim)

    def reset_parameters(self):
        # these bounds are based on fan_in, so it's the size of the previous layer
        bound2 = 1 / ((self.layer_size[0])**0.5)
        nn.init.uniform_(self.bias2, -bound2, bound2)

        bound3 = 1 / (self.layer_size[1]**0.5)
        nn.init.uniform_(self.bias3, -bound3, bound3)

        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        layer1 = get_embeddings(self.emb1)
        layer2 = get_embeddings(self.emb2)
        layer3 = get_embeddings(self.emb3)

        weight1 = unscaled_dot(layer1, layer2)
        weight2 = unscaled_dot(layer2, layer3)

        pos_enc = embedding.positional_encoding2d(self.n_dims, 28, 28).view(self.n_dims, -1).permute(1, 0).to(self.dev_param.device)

        out = x.view(x.shape[0], -1)
        out = pos_enc + out
        out = F.relu(torch.matmul(out, weight1) + self.bias2)
        out = torch.matmul(out, weight2) + self.bias3
        return out



'''
"Small" models, for use with the simple binary classifier dataset
'''

class FCSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_size = [2, 100, 100, 1]

        # Submodules
        self.fc1 = nn.Linear(self.layer_size[0], self.layer_size[1])
        self.fc2 = nn.Linear(self.layer_size[1], self.layer_size[2])
        self.fc3 = nn.Linear(self.layer_size[2], self.layer_size[3])

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

class NeuronEmbeddingSmall(nn.Module):
    def __init__(self):
        super(NeuronEmbeddingSmall, self).__init__()
        self.n_dims = 8
        self.layer_size = [2, 100, 100, 1]

        self.biases = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.layer_size[1])),
            nn.Parameter(torch.Tensor(self.layer_size[2])),
            nn.Parameter(torch.Tensor(self.layer_size[3]))
        ])
        self.reset_parameters()
        
        self.embs = nn.ModuleList([
            nn.Embedding(num_embeddings = self.layer_size[0], embedding_dim = self.n_dims),
            nn.Embedding(num_embeddings = self.layer_size[1], embedding_dim = self.n_dims),
            nn.Embedding(num_embeddings = self.layer_size[2], embedding_dim = self.n_dims),
            nn.Embedding(num_embeddings = self.layer_size[3], embedding_dim = self.n_dims)
        ])

    def reset_parameters(self):
        for i, param in enumerate(self.biases):
            init_bias(param, self.layer_size[i])

        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, return_intermediate = False):
        weights = get_weights(self.embs)

        out0 = x.view(x.shape[0], -1)
        out1 = F.relu(out0 @ weights[0] + self.biases[0])
        out2 = F.relu(out1 @ weights[1] + self.biases[1])
        out3 = out2 @ weights[2] + self.biases[2]
        if return_intermediate:
            return out1, out2, out3
        return out3

class FCSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_size = [2, 100, 100, 1]

        # Submodules
        self.fc1 = nn.Linear(self.layer_size[0], self.layer_size[1])
        self.fc2 = nn.Linear(self.layer_size[1], self.layer_size[2])
        self.fc3 = nn.Linear(self.layer_size[2], self.layer_size[3])

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)