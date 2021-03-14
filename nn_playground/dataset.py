from sklearn import datasets
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import logging

class MyData(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
        ss = StandardScaler()
        self.X = ss.fit_transform(self.X)
        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.int)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array(self.y[idx]))

    def plot_data(self, filename):
        logging.debug(F'Data plot X: {self.X.shape}; y: {self.y.shape}')
        plt.scatter(self.X[:,0], self.X[:,1], c=self.y)
        plt.savefig(filename)

class Circles(MyData):
    def __init__(self, n_samples, shuffle, noise, random_state=0, factor=.8):
        X, y = datasets.make_circles(n_samples=n_samples, 
            shuffle=shuffle, noise=noise,
            random_state=random_state,
            factor=factor)

        super(Circles,self).__init__(X, y)

class Gussian(MyData):
    def __init__(self, n_samples, shuffle, mean=None, cov=1.0, n_features=2, n_classes=2, random_state=0):
        X,y = datasets.make_gaussian_quantiles(mean=mean,
            cov=cov,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            shuffle=shuffle,
            random_state=random_state)
        
        super(Gussian,self).__init__(X, y)

class Moons(MyData):
    def __init__(self, n_samples, shuffle, noise, random_state=0):
        X,y = datasets.make_moons(n_samples=n_samples,
            shuffle=shuffle,
            noise = noise,
            random_state=random_state)

        super(Moons,self).__init__(X, y)

class Custom(MyData):
     def __init__(self, filename, sep='\t', header=None):
        data = pd.read_csv(filename, sep=sep, header=header)
        if data.shape[1]!=3:
            logging.error('Incorrect input data shape')

        X,y = data.iloc[:,:2].to_numpy(),data.iloc[:,2].to_numpy()
        super(Custom,self).__init__(X, y)