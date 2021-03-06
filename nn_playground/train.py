import json
import logging
from datetime import datetime

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import dataset
from visualize_utils import make_meshgrid, predict_proba_on_mesh, plot_predictions

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

        # Select device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # Initialize model
        self.model = model.to(self.device)

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

        # create tensorboard output
        self.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter("logs/"+self.experiment_name)

    def fit(self, train_dataloader, n_epochs):
        """ Train model """
        self.model.train()
        
        first_batch = next(iter(train_dataloader))
        self.writer.add_graph(self.model, first_batch[0])

        n_iter = 0
        for epoch in range(n_epochs):
            logging.debug(F"epoch: {epoch}")
            epoch_loss = 0
            for x_batch, y_batch in train_dataloader:
                self.optimizer.zero_grad()
                output = self.model(x_batch.to(self.device))
                loss=self.criterion(output,y_batch.to(self.device))
                loss.backward()
                self.optimizer.step()
                epoch_loss+=loss.item()

                # Summary plots
                self.writer.add_scalar('Loss on epoch end', epoch_loss, n_iter)
                n_iter+=1

        logging.debug("Fit completed. Model parameters:")
        for param in self.model.parameters():
            logging.debug(param)

    def predict_proba(self, test_dataloader):
        all_outputs = torch.tensor([], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in test_dataloader:
                output = self.model(x_batch.to(self.device)).to(torch.device("cpu"))
                all_outputs = torch.cat((all_outputs,output),0)
        return all_outputs

    def predict(self, test_dataloader):
        output_proba = self.predict_proba(test_dataloader)
        return torch.max(output_proba.data,1)[1]

    def predict_proba_tensor(self,T):
        self.model.eval()
        with torch.no_grad():
            output=self.model(T)
        return output

def generate_dataset(data_coniguration):
    data = None
    datatype = data_coniguration['type']
    if datatype == 'circles':
        data=dataset.Circles(n_samples = data_coniguration['n_samples'],
            shuffle = data_coniguration['shuffle'],
            noise = data_coniguration['noise'],
            random_state = data_coniguration['random_state'])
    elif datatype == 'gaussian':
        data=dataset.Gussian(n_samples = data_coniguration['n_samples'],
            shuffle = data_coniguration['shuffle'],
            random_state = data_coniguration['random_state'])
    elif datatype == 'moons':
        data=dataset.Moons(n_samples = data_coniguration['n_samples'], 
            shuffle = data_coniguration['shuffle'],
            noise = data_coniguration['noise'],
            random_state = data_coniguration['random_state'])
    elif datatype == 'custom':
        data=dataset.Custom('custom_dataset.csv')
    else:
        logging.error(F'Incorrect type of dataset: {datatype}')
    
    return data

def predict_proba_on_mesh_tensor(trainer, xx, yy):
    q = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    Z = trainer.predict_proba_tensor(q)[:,1]
    Z = Z.reshape(xx.shape)
    return Z

if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    # read model and trainer configuration
    with open('train_coniguration.json') as json_file:
        train_coniguration = json.load(json_file)
        layers = train_coniguration['layers']
        batch_size = train_coniguration['batch_size']
        optimizer = train_coniguration['optimizer']
        learning_rate = train_coniguration['learning_rate']
        criterion = train_coniguration['criterion']
        n_epochs = train_coniguration['n_epochs']

    # read data configuration
    with open('data_coniguration.json') as json_file:
        data_coniguration = json.load(json_file)

    # generate train data set
    train_set = generate_dataset(data_coniguration['train'])
    train_set.plot_data('output/train_set.png')
    train_dataloader=DataLoader(train_set, batch_size=batch_size, shuffle = True)

    # train model
    model=FC(layers)
    trainer = Trainer(model, learning_rate, optimizer, criterion)
    trainer.fit(train_dataloader, n_epochs)

    # test model
    test_set = generate_dataset(data_coniguration['test'])
    test_set.plot_data('output/test_set.png')
    test_dataloader=DataLoader(test_set, batch_size=batch_size, shuffle = False)

    test_prediction = trainer.predict(test_dataloader)
    test_prediction_proba = trainer.predict_proba(test_dataloader)
    
    # visualize decision surface
    X_train, y_train = train_set.get_numpy_data()
    X_test, y_test = test_set.get_numpy_data()

    xx,yy = make_meshgrid(X_train, X_test)
    Z=predict_proba_on_mesh_tensor(trainer, xx, yy)
    plot_predictions(xx, yy, Z,
        filename="output/plot_predictions_train.png",
        X=X_train,
        y=y_train)

    plot_predictions(xx, yy, Z,
        filename="output/plot_predictions_test.png",
        X=X_test,
        y=y_test)