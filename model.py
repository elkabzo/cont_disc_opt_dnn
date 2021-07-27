import numpy as np
import torch
from torch import nn as nn, optim as optim

class Network(nn.Module):

    def __init__(self,train_loader,test_loader,layers_size=[28*28,50,50,10],
                 costum_weights=None, init_mode='xavier', deviation=1, activation = 'Linear',
                 print_progress=False, criterion=nn.CrossEntropyLoss(), label = 'network', batch_size_train = 100):

        # call constructor from superclass
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #train and test loaders
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size_train = batch_size_train

        #data for visualization
        self.label = label
        self.train_time_stamps = []
        self.train_losses = []
        self.train_end2end = []
        self.train_weights = []
        self.train_weights2 = []
        # self.test_losses = []
        # self.test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

        #initialize input arguments
        self.criterion = criterion
        self.activation = activation
        self.layers_size = layers_size
        self.initial_weights = []
        if (costum_weights is not None):
            self.initial_weights = costum_weights

        # initialize weights
        if activation == 'conv_max_pool':
            self.pad = torch.nn.ZeroPad2d(2)
            self.conv1 = nn.Conv2d(1,6,5, bias=True)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6,16,5,bias=True)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.layers_size = [16*5*5,120,84,10]
        if activation == 'conv_subsample':
            self.pad = torch.nn.ZeroPad2d(2)
            self.conv1 = nn.Conv2d(1,6,5,stride=2,bias=False)
            self.conv2 = nn.Conv2d(6,16,5,stride=2,bias=False)
            self.layers_size = [16*5*5,120,84,10]
        self.initialize_linear_functions()
        self.set_weights(costum_weights, init_mode, deviation)

        # prints
        self.print_progress = print_progress
        if self.print_progress: print("--------------------------\ninit deep network of size: ", layers_size)

        # initialize training information
        self.epoch = 1

    def initialize_linear_functions(self):
        self.relu = nn.ReLU(inplace=True).to(self.device)
        self.linear_functions = nn.ModuleList().to(self.device)
        with torch.no_grad():
            for i, (prev_layer_size, next_layer_size) in enumerate(zip(self.layers_size[:-1], self.layers_size[1:])):
                bias = True if (self.activation == 'conv_max_pool') else False
                #bias = True if ((i == 0 and self.activation!='conv_subsample') or self.activation == 'conv_max_pool') else False
                self.linear_functions.append(nn.Linear(prev_layer_size, next_layer_size,bias=bias).to(self.device)).to(self.device)

    def set_weights(self, costum_weights=None, init_mode='gaussian', deviation=1):
        with torch.no_grad():
            for i, (prev_layer_size, next_layer_size) in enumerate(zip(self.layers_size[:-1], self.layers_size[1:])):
                if (costum_weights is not None):
                    # print("\nsetting costum weights\n")
                    self.linear_functions[i].weight.copy_(costum_weights[i])
                elif init_mode == 'gaussian':
                    torch.nn.init.normal_(self.linear_functions[i].weight.to(self.device), mean=0.0,
                                          std=deviation / np.sqrt(prev_layer_size)).to(self.device)
                    self.initial_weights.append(self.linear_functions[i].weight.to(self.device))
                else:  # init_mode=='xavier'
                    torch.nn.init.xavier_uniform_(self.linear_functions[i].weight.to(self.device)).to(self.device)
                    self.initial_weights.append(self.linear_functions[i].weight.to(self.device))

    def forward(self, x):
        if self.activation == 'conv_max_pool':
            x = self.pad(x)
            x = self.pool1(self.relu(self.conv1(x)))
            x = self.pool2(self.relu(self.conv2(x)))
        if self.activation == 'conv_subsample':
            x = self.pad(x)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
        x = x.view(self.batch_size_train, -1)
        for i, function in enumerate(self.linear_functions):
            x = function(x)
            if (self.activation == 'ReLU' or self.activation == 'relu' or self.activation == 'conv_max_pool' or self.activation == 'conv_subsample') and i < len(self.layers_size)-2:
                x = self.relu(x)
        return x

    def continue_training(self):
        self.train()

    def train(self):
        for epoch in range(self.epoch, self.n_epochs + 1):
            if epoch % 10000 == 0:
                self.epoch = epoch
                self.save_state()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self(data).to(self.device)
                loss = self.criterion(output, target).to(self.device)
                if epoch == 1:
                    print('Train Epoch: {} [({:.0f}%)]\tLoss: {:.6f}'.format(0, 0, loss.item()))
                    self.train_time_stamps.append(0)
                    self.train_losses.append(loss.item())
                    self.train_weights.append(self.weights_tensor())
                    #self.train_end2end.append(self.end_2_end())
                loss.backward()
                self.optimizer.step()
                if epoch % (self.n_epochs / self.nm_of_time_stamps) == 0:
                    print(
                        'Train Epoch: {} [({:.0f}%)]\tLoss: {:.6f}'.format(epoch, 100. * epoch / self.n_epochs, loss.item()))
                    self.train_time_stamps.append(epoch)
                    self.train_losses.append(loss.item())
                    self.train_weights.append(self.weights_tensor())
                    #self.train_end2end.append(self.end_2_end())
        self.epoch = self.n_epochs + 1
        self.save_state()

    def init_train_params(self, n_epochs = 100 , learning_rate = 0.01 , nm_of_time_stamps = 100 , lr_ratio_from_base = 1):
        # initialize train data
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        self.nm_of_time_stamps = nm_of_time_stamps
        self.lr_ratio_from_base = lr_ratio_from_base

    def test_acc(self):

        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                images, labels = data
                labels = labels
                images = torch.flatten(images, start_dim=1)
                output = self.forward(images)


                # loss
                loss = self.criterion(output, labels)
                total_loss += loss.item()

        return (total_loss / len(self.test_loader))

    def save_state(self, file_name = None):

        if file_name is None:
            file_name = self.label + '_model.pth'

        #torch.save(self.state_dict(), file_name)
        torch.save(self, file_name)

        # torch.save(self.optimizer.state_dict(), 'results/optimizer.pth')

    """
    def load_state(self, file_name = None):

        if file_name is None:
            file_name = self.label + '_model.pth'

        network_dict = torch.load(file_name)
        self.load_state_dict(network_dict)

        # optimizer_dict = torch.load('results/optimizer.pth')
        # self.optimizer.load_state_dict(optimizer_dict)
    """

    def end_2_end(self):
        for i, function in enumerate(self.linear_functions):
            if i==0:
                product = self.linear_functions[0].weight
            else:
                product = torch.mm(self.linear_functions[i].weight,product)
        return product

    def weights_tensor(self):
        weights_tensor = torch.Tensor([])
        if self.activation == 'conv_max_pool' or self.activation == 'conv_subsample':
            weights_tensor = torch.cat([weights_tensor,self.conv1.weight.view(-1)])
            weights_tensor = torch.cat([weights_tensor, self.conv2.weight.view(-1)])
        for i, function in enumerate(self.linear_functions):
            weights_tensor = torch.cat([weights_tensor,self.linear_functions[i].weight.view(-1)])
        return weights_tensor