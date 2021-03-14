import torch
from torch import nn
import dataset

import json
import logging
class FC(nn.Module):
    """Fully-connected NN model"""
    def __init__(self, layers):
        """Initialize fully-connected NN with layers"""
        super(FC, self).__init__()
        modules = []
        for layer in layers:
            in_features=layer['in_features']
            out_features=layer['out_features']
            activation = layer['activation']
            modules.append(nn.Linear(in_features,out_features))
            if activation == "sigmoid":
                modules.append(nn.Sigmoid())
            elif activation == "relu":
                modules.append(nn.ReLU())
            elif activation == "tanh":
                modules.append(nn.Tanh())
            elif activation == "rrelu":
                modules.append(nn.RReLU())
            else:
                logging.error('Unknown activation function')
                return

        modules.append(nn.Softmax())
        self.net = nn.Sequential(*modules)
        logging.debug(F'The model is initialized :\n{self.net}')
        for param in self.net.parameters():
            logging.debug(param)

    def forward(self, x):
        """Forward pass"""
        if not self.net:
            logging.error('The model is not initialized properly')
            return None
        return self.net(x)

if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    with open('train_coniguration.json') as json_file:
        train_coniguration = json.load(json_file)
        layers = train_coniguration['layers']
        batch_size = train_coniguration['batch_size']
        optimizer = train_coniguration['optimizer']
        learning_rate = train_coniguration['learning_rate']
    
    FC(layers)

    with open('data_coniguration.json') as json_file:
        data_coniguration = json.load(json_file)
        datatype=data_coniguration['type']

        if datatype == 'circles':
            ds=dataset.Circles(n_samples = data_coniguration['n_samples'],
                shuffle = data_coniguration['shuffle'],
                noise = data_coniguration['noise'])
        elif datatype == 'gaussian':
            ds=dataset.Gussian(n_samples = data_coniguration['n_samples'],
                shuffle = data_coniguration['shuffle'])
        elif datatype == 'moons':
            ds=dataset.Moons(n_samples = data_coniguration['n_samples'], 
                shuffle = data_coniguration['shuffle'],
                noise = data_coniguration['noise'])
        elif datatype == 'custom':
            ds=dataset.Custom('custom_dataset.csv')
        else:
            logging.error(F'Incorrect type of dataset: {datatype}')

    ds.plot_data('output/plot.png')