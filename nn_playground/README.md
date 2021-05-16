Getting Started

1. Build docker container

```docker build /path/to/directory/nn_playground -t nn```

2. Provide network configuration and train parameters in train_coniguration.json

Options:


```
layers: list of layers for the fully connected network, for each layer input feature number (in_features), output feature number (out_features) and the type of activation function (activation) should be specified. Possible values of activation: sigmoid, relu, tanh, rrelu, elu.

batch_size: batch size

optimizer: optimizer for training, possible values: sgd, adam, adagrad, rmsprop

learning_rate: training learning rate

criterion: The criterion for training, possible values: crossentropyloss, hingeembeddingloss

n_epochs: number of epoches
```

3. Provide dataset configuration in data_coniguration.json. Train and test datasets should be defined:

Options:


```
type: sklearn dataset type, possible values: circles, gaussian, moons, custom

Data for a custom dataset is taken from custom_dataset.csv.

n_samples: number of samples

shuffle: whether to shuffle the dataset

noise: data noise

random_state: random state for data generation
```

4. Run the docker container to train the network and test it on the test dataset. The output folder and the log folder should be mapped to the docker container /output and /logs folders.


```docker run  -v /path/to/output:/output -v /path/to/logs:/logs nn python train.py```

5. Check the result files in the output folder:

```
train_set.png - plot for the train dataset

test_set.png - plot for the test dataset

plot_predictions.png - prediction surfaces of the classifier
```