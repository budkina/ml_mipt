import json
import logging
from datetime import datetime

import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import dataset

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
            elif activation == "elu":
                modules.append(nn.ELU())               
            else:
                logging.error('Unknown activation function')
                return

        modules.append(nn.Softmax(dim=1))
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

class Trainer:
    def __init__(self, model, learning_rate, optimizer=None, criterion=None):
        # Initialize model
        self.model = model

        # Select optimizer
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer == 'adagrad':
            self.optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            logging.error('Incorrect optimizer value')
            return

        # Select loss function
        if criterion == 'crossentropyloss':
            self.criterion = nn.CrossEntropyLoss()
        elif criterion == 'hingeembeddingloss':
            self.criterion = nn.HingeEmbeddingLoss()
        else:
            logging.error('Incorrect loss function')
            return

        # Select device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # create tensorboard output
        self.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter("logs/"+self.experiment_name)

    def fit(self, train_dataloader, n_epochs):
        """ Train model """
        self.model.train()
        
        first_batch = next(iter(train_dataloader))
        self.writer.add_graph(self.model, first_batch[0])

        for epoch in range(n_epochs):
            print("epoch: ", epoch)
            epoch_loss = 0
            for x_batch, y_batch in train_dataloader:
                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss=self.criterion(output,y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss+=loss.item()

                # Summary plots
                self.writer.add_scalar('Loss on epoch end', epoch_loss)
                self.writer.add_scalar('Loss/train size', epoch_loss/len(train_dataloader))

        logging.debug("Fit completed. Model parameters:")
        for param in self.model.parameters():
            logging.debug(param)


    def predict(self, test_dataloader):
        all_outputs = torch.tensor([], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                pass
        return all_outputs

    def predict_proba(self, test_dataloader):
        all_outputs = torch.tensor([], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                pass
        return all_outputs

    def predict_proba_tensor(self, T):
        self.model.eval()
        with torch.no_grad():
            pass
        return output

if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    with open('train_coniguration.json') as json_file:
        train_coniguration = json.load(json_file)
        layers = train_coniguration['layers']
        batch_size = train_coniguration['batch_size']
        optimizer = train_coniguration['optimizer']
        learning_rate = train_coniguration['learning_rate']
        criterion = train_coniguration['criterion']
        n_epochs = train_coniguration['n_epochs']

    with open('data_coniguration.json') as json_file:
        data_coniguration = json.load(json_file)
        datatype=data_coniguration['type']

        if datatype == 'circles':
            train_set=dataset.Circles(n_samples = data_coniguration['n_samples'],
                shuffle = data_coniguration['shuffle'],
                noise = data_coniguration['noise'])
        elif datatype == 'gaussian':
            train_set=dataset.Gussian(n_samples = data_coniguration['n_samples'],
                shuffle = data_coniguration['shuffle'])
        elif datatype == 'moons':
            train_set=dataset.Moons(n_samples = data_coniguration['n_samples'], 
                shuffle = data_coniguration['shuffle'],
                noise = data_coniguration['noise'])
        elif datatype == 'custom':
            train_set=dataset.Custom('custom_dataset.csv')
        else:
            logging.error(F'Incorrect type of dataset: {datatype}')
            sys.exit()

    train_set.plot_data('output/plot.png')
    train_dataloader=DataLoader(train_set, batch_size=batch_size, shuffle = True)
    model=FC(layers)
    trainer = Trainer(model, learning_rate, optimizer, criterion)
    trainer.fit(train_dataloader, n_epochs)