import torch
import torch.nn as nn
import torch.nn.functional as F
import embedding
from itertools import chain

def get_embeddings(emb):
    '''
    Helper function to get all the embeddings for a layer.
    Technically we could just grab the weight tensor directly, but going through the embedding layer enables us to use some options
    '''
    return emb(torch.LongTensor([i for i in range(emb.num_embeddings)]).to(emb.weight.device))

def scaled_dot(layer1, layer2):
    '''
    Scaled dot product, like the one used in Transformer.
    Since we multiply two embeddings our scaling factor is squared (canceling out the sqrt)
    '''
    return torch.matmul(layer1, layer2.transpose(0, 1))/(layer1.shape[-1]) #**0.5)

def unscaled_dot(layer1, layer2):
    '''
    Unscaled (vanilla) dot product
    '''
    return torch.matmul(layer1, layer2.transpose(0, 1))

def get_layers(embs):
    '''
    Gets the embeddings for all neurons in a layer.
    
    Input: list of embedding modules
    Output: list of tensors (embeddings)
    '''
    return [get_embeddings(emb) for emb in embs]


def get_weights(layers, env = None):
    '''
    Calculates the weight matrices by calculating dot product alignment between consecutive pairs of layers.
    I.e., if the input is [layer1, layer2, layer3] we calculate [dot(layer1, layer2), dot(layer2, layer3)]
    
    Input: layers: list of layers, where each layer is represented as the tensor of neuron embeddings
           env:    optional environment function (probably a linear feedforward net or similar)
    Output: list of weight tensors
    '''
    if env is None:
        return [unscaled_dot(l1, l2) for l1,l2 in zip(layers[:-1], layers[1:])]
    else:
        return [unscaled_dot(env(l1), env(l2)) for l1,l2 in zip(layers[:-1], layers[1:])]

class FCNet(nn.Module):
    '''
    A simple fully connected model.
    '''
    def __init__(self):
        super(FCNet, self).__init__()
        self.layer_size = [28*28, 400, 10]

        self.fc1 = nn.Linear(self.layer_size[0], self.layer_size[1])
        self.fc2 = nn.Linear(self.layer_size[1], self.layer_size[2])

    def forward(self, x):
        out = F.relu(self.fc1(x.view(x.shape[0], -1)))
        return self.fc2(out)

# What if we also add the neuron embeddings to the input?

class NeuronEmbedding(nn.Module):
    '''
    Fully connected model using neuron embeddings.
    '''
    def __init__(self, n_dims = 64):
        super().__init__()
        self.n_dims = n_dims
        self.layer_size = [28*28, 400, 10]

        self.biases = nn.ParameterList([
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
            # these bounds are based on fan_in, so it's the size of the previous layer
            bound = 1 / ((self.layer_size[i])**0.5)
            nn.init.uniform_(param, -bound, bound)

        for i, emb in enumerate(self.embs):
            # this rescaled initialization SHOULD remove the need for scaled_dot
            nn.init.normal_(emb.weight, std=1.0/(self.n_dims**0.5))


    def forward(self, x, return_intermediate = False):
        layers = get_layers(self.embs)
        weights = get_weights(layers)

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
    def __init__(self):
        super().__init__()
        self.layer_size = [1, 10, 40]
        self.conv1 = nn.Conv2d(self.layer_size[0], self.layer_size[1], (3, 3))
        self.conv2 = nn.Conv2d(self.layer_size[1], self.layer_size[2], (3, 3))
        self.maxpool = nn.MaxPool2d((2,2))
        self.lin1 = nn.Linear(1000, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1) # flatten out channel, w, h into one dimension
        out = self.lin1(x)
        return out

class NeuronEmbeddingConv(nn.Module):
    '''
    Convolutional variant using neuron embeddings.
    '''
    def __init__(self, n_dims = 32):
        super().__init__()
        self.n_dims = n_dims
        self.kernel_size = 3
        self.conv_layer_size = [1, 10, 40]
        self.lin_layer_size = [1000, 10]

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
            # these bounds are based on fan_in, so it's the size of the previous layer
            bound = 1 / ((self.conv_layer_size[i])**0.5)
            nn.init.uniform_(param, -bound, bound)

        for i, param in enumerate(self.lin_biases):
            # these bounds are based on fan_in, so it's the size of the previous layer
            bound = 1 / ((self.lin_layer_size[i])**0.5)
            nn.init.uniform_(param, -bound, bound)

        for i, emb in enumerate(chain(self.field_embs, self.conv_embs, self.lin_embs)):
            # this rescaled initialization SHOULD remove the need for scaled_dot
            nn.init.normal_(emb.weight, std=1.0/(self.n_dims**0.5))


    def forward(self, x, return_intermediate = False):
        # build the embedding pieces
        conv_layers = get_layers(self.conv_embs)
        conv_layers_reshaped = [t.reshape((-1, 1, 1, self.n_dims)) for t in conv_layers]
        conv_fields = get_layers(self.field_embs)
        conv_fields_reshaped = [t.reshape((-1, self.kernel_size, self.kernel_size, self.n_dims)) for t in conv_fields]

        # build compound embedding
        conv_output_embs_0 = conv_layers_reshaped[1] + conv_fields_reshaped[0]
        conv_output_embs_1 = conv_layers_reshaped[2] + conv_fields_reshaped[1]

        # calculate the weights
        conv_weight_0 = conv_output_embs_0 @ conv_layers[0].T
        conv_weight_0 = conv_weight_0.permute(0, 3, 1, 2).contiguous()

        conv_weight_1 = conv_output_embs_1 @ conv_layers[1].T
        conv_weight_1 = conv_weight_1.permute(0, 3, 1, 2).contiguous()


        lin_layers = get_layers(self.lin_embs)
        lin_weights = get_weights(lin_layers)
        
        # actual forward step
        out0 = x
        out1 = F.relu(F.conv2d(out0, weight=conv_weight_0, bias=self.conv_biases[0]))
        out1 = self.maxpool(out1)

        out2 = F.relu(F.conv2d(out1, weight=conv_weight_1, bias=self.conv_biases[1]))
        out2 = self.maxpool(out2).contiguous()

        out3 = out2.view(out2.shape[0], -1) # flatten out channel, w, h into one dimension

        out4 = out3 @ lin_weights[0] + self.lin_biases[0]
        if return_intermediate:
            return out1, out2, out3, out4
        return out4

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
