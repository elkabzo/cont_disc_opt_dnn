# -*- coding: utf-8 -*-

#@title imports & settings { form-width: "10%" }
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
from matplotlib import lines as plt_lines
from matplotlib import markers as plt_markers

#@title Data load { form-width: "10%" }
def data_load(batch_size_train = 1000 , batch_size_test = 1000):

    trainset = torchvision.datasets.MNIST(  'files/', train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                          ]))
    if batch_size_train <= 1001:
        trainset = torch.utils.data.Subset(trainset, list(range(0, batch_size_train*50,50)))
    else:
        trainset = torch.utils.data.Subset(trainset, list(range(0, batch_size_train)))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=False, num_workers = 0)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST( 'files/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                   ])),
                                    batch_size=batch_size_test, shuffle=True)

    return train_loader,test_loader

#@title neural network { form-width: "10%" }
class Network(nn.Module):

    def __init__(self,train_loader,test_loader,layers_size=[28*28,50,50,10],
                 costum_weights=None, init_mode='xavier', deviation=1, activation = 'Linear',
                 print_progress=False, criterion=nn.CrossEntropyLoss(), label = 'network', batch_size_train = 100):

        # call constructor from superclass
        super().__init__()

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
        self.relu = nn.ReLU(inplace=True).to(device)
        self.linear_functions = nn.ModuleList().to(device)
        with torch.no_grad():
            for i, (prev_layer_size, next_layer_size) in enumerate(zip(self.layers_size[:-1], self.layers_size[1:])):
                bias = True if (self.activation == 'conv_max_pool') else False
                #bias = True if ((i == 0 and self.activation!='conv_subsample') or self.activation == 'conv_max_pool') else False
                self.linear_functions.append(nn.Linear(prev_layer_size, next_layer_size,bias=bias).to(device)).to(device)

    def set_weights(self, costum_weights=None, init_mode='gaussian', deviation=1):
        with torch.no_grad():
            for i, (prev_layer_size, next_layer_size) in enumerate(zip(self.layers_size[:-1], self.layers_size[1:])):
                if (costum_weights is not None):
                    # print("\nsetting costum weights\n")
                    self.linear_functions[i].weight.copy_(costum_weights[i])
                elif init_mode == 'gaussian':
                    torch.nn.init.normal_(self.linear_functions[i].weight.to(device), mean=0.0,
                                          std=deviation / np.sqrt(prev_layer_size)).to(device)
                    self.initial_weights.append(self.linear_functions[i].weight.to(device))
                else:  # init_mode=='xavier'
                    torch.nn.init.xavier_uniform_(self.linear_functions[i].weight.to(device)).to(device)
                    self.initial_weights.append(self.linear_functions[i].weight.to(device))

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
                data = data.to(device)
                target = target.to(device)
                self.optimizer.zero_grad()
                output = self(data).to(device)
                loss = self.criterion(output, target).to(device)
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

#@title graphs { form-width: "10%" }
def plot_networks_loss_graph(base_network , networks_to_compare = [], download = False, networks_type_str = 'Regular', scale = 'regular'):
    #figure settings
    #plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({'legend.fontsize': 9})
    plt.figure(figsize=(3.2, 3))
    plt.ylabel('Training Loss')
    plt.xlabel('$\eta_{0}$ Iterations')

    #plot
    plt.plot(base_network.train_time_stamps, base_network.train_losses, label='$\eta_{0}$')
    for i,network in enumerate(networks_to_compare):
        plt.plot(base_network.train_time_stamps, network.train_losses, label='$\eta_{0}/$'+str(network.lr_ratio_from_base))

    plt.legend()

    if scale == 'log':
        plt.yscale('log')

    if networks_type_str == 'Linear':
        plt.title('Fully Connected, Linear Activation', pad=10)
    if networks_type_str == 'ReLU':
        plt.title('Fully Connected, ReLU Activation', pad=10)
    if networks_type_str == 'conv_subsample':
        plt.title('Convolutional, ReLU Activation, No Pooling', pad=10)
    if networks_type_str == 'conv_max_pool':
        plt.title('Convolutional, ReLU Activation, Max Pooling', pad=10)

    plt.xlim(left=0,right=base_network.train_time_stamps[-1])
    #plt.tight_layout(pad=0.05)
    plt.tight_layout()
    if download == True:
        if scale == 'log':
            plt.savefig('LossGraph'+networks_type_str+'_log_scale.pdf',bbox_inches = "tight")
        else:
            plt.savefig('LossGraph'+networks_type_str+'.pdf',bbox_inches = "tight")
        #files.download('LossGraph'+networks_type_str+'.pdf')
    plt.show()

def plot_networks_distance_graph(base_network, networks_to_compare = [], download = False, networks_type_str = 'Regular', scale = 'regular'):
    #figure settings
    #plt.rcParams.update({'font.size': 15})
    plt.rcParams.update({'legend.fontsize': 9})
    plt.figure(figsize=(3.2, 3))
    plt.ylabel('Distance')
    plt.xlabel('$\eta_{0}$ Iterations')

    #initialize lists for plot
    base_network_norms = []
    networks_distance = []
    for compare_network in networks_to_compare:
        networks_distance.append([])

    #insert values into lists
    for i,train_weight in enumerate(base_network.train_weights):
        base_network_norms.append(torch.norm(train_weight - base_network.train_weights[0]).cpu().detach().numpy())
        for j,compare_network in enumerate(networks_to_compare):
            networks_distance[j].append(torch.norm(base_network.train_weights[i]-compare_network.train_weights[i]).cpu().detach().numpy())

    #plot
    plt.plot(base_network.train_time_stamps[0:],base_network_norms[0:],label='$\eta_{0}$ from init')
    for j,compare_network in enumerate(networks_to_compare):
        plt.plot(base_network.train_time_stamps[0:],networks_distance[j][0:],label='$\eta_{0}$ from $\eta_{0}/$'+str(compare_network.lr_ratio_from_base))
    plt.legend()
    if scale == 'log':
        plt.yscale('log')

    if networks_type_str == 'Linear':
        plt.title('Fully Connected, Linear Activation', pad=10)
    if networks_type_str == 'ReLU':
        plt.title('Fully Connected, ReLU Activation', pad=10)
    if networks_type_str == 'conv_subsample':
        plt.title('Convolutional, ReLU Activation, No Pooling', pad=10)
    if networks_type_str == 'conv_max_pool':
        plt.title('Convolutional, ReLU Activation, Max Pooling', pad=10)

    plt.xlim(left=0,right=base_network.train_time_stamps[-1])
    #plt.tight_layout(pad=0.05)
    plt.tight_layout()
    if download == True:
        if scale == 'log':
            plt.savefig('DistGraph'+networks_type_str+'_log_scale.pdf',bbox_inches = "tight")
        else:
            plt.savefig('DistGraph'+networks_type_str+'.pdf',bbox_inches = "tight")
        #files.download('DistGraph'+networks_type_str+'.pdf')
    plt.show()

def set_networks_loss_subgraph(base_network , subgraph, networks_to_compare = [], download = False, networks_type_str = 'Regular', scale = 'regular',axis_size=14.5):
    subgraph.set_ylabel('Training Loss',fontsize=axis_size)
    subgraph.set_xlabel('$\eta_{0}$ Iterations',fontsize=axis_size)

    lines = list(plt_lines.lineStyles.keys())
    markers = list(plt_markers.MarkerStyle.markers.keys())
    #lines = [lines[i] for i in [0,1,3]]
    lines = [lines[0],lines[0],lines[1],lines[1],lines[3]]
    markers = [markers[i] for i in [2, 3, 4, 12, 14, 21, 24]]
    markers = [markers[3],markers[0],markers[1],markers[2],markers[4],markers[5],markers[6]]

    #plot
    subgraph.plot(base_network.train_time_stamps, base_network.train_losses, label='$\eta_{0}$',linestyle = (0,(1,1)), marker=markers[0],markevery=10,markersize=7)#linewidth = 3
    for i,network in enumerate(networks_to_compare):
        subgraph.plot(base_network.train_time_stamps, network.train_losses, label='$\eta_{0}/$'+str(network.lr_ratio_from_base),linestyle = (0.25*(i+1),(1,1)), marker=markers[(1+i)%len(markers)],markevery=10)

    subgraph.legend()

    if scale == 'log':
        subgraph.set_yscale('log')

    subgraph.set_xlim(left=0,right=base_network.train_time_stamps[-1])
    subgraph.set_ylim(bottom = 0)
    return subgraph

def set_networks_distance_subgraph(base_network, subgraph, networks_to_compare = [], download = False, networks_type_str = 'Regular', scale = 'regular',axis_size=14.5):
    subgraph.set_ylabel('Distance',fontsize=axis_size)
    subgraph.set_xlabel('$\eta_{0}$ Iterations',fontsize=axis_size)

    lines = list(plt_lines.lineStyles.keys())
    lines = [lines[i] for i in [0,1,3]]
    markers = list(plt_markers.MarkerStyle.markers.keys())
    markers = [markers[i] for i in [2, 3, 4, 12, 14, 21, 24]]
    markers = [markers[3],markers[0],markers[1],markers[2],markers[4],markers[5],markers[6]]

    #initialize lists for plot
    base_network_norms = []
    networks_distance = []
    for compare_network in networks_to_compare:
        networks_distance.append([])

    #insert values into lists
    for i,train_weight in enumerate(base_network.train_weights):
        base_network_norms.append(torch.norm(train_weight - base_network.train_weights[0]).cpu().detach().numpy())
        for j,compare_network in enumerate(networks_to_compare):
            networks_distance[j].append(torch.norm(base_network.train_weights[i]-compare_network.train_weights[i]).cpu().detach().numpy())

    #plot
    #subgraph.plot(base_network.train_time_stamps[0:],base_network_norms[0:],label='$\eta_{0}$ from init', linestyle = lines[0], marker=markers[0],markevery=10,markersize=7)
    subgraph.plot(base_network.train_time_stamps[0:],base_network_norms[0:],label='$\eta_{0}$ from init', linestyle = (0,(1,1)), marker=markers[0],markevery=10,markersize=7)
    for j,compare_network in enumerate(networks_to_compare):
        #subgraph.plot(base_network.train_time_stamps[0:],networks_distance[j][0:],label='$\eta_{0}$ from $\eta_{0}/$'+str(compare_network.lr_ratio_from_base),linestyle = lines[(1+j)%len(lines)], marker=markers[(1+j)%len(markers)],markevery=10)
        subgraph.plot(base_network.train_time_stamps[0:],networks_distance[j][0:],label='$\eta_{0}$ from $\eta_{0}/$'+str(compare_network.lr_ratio_from_base),linestyle = (0.25*(j+1),(1,1)), marker=markers[(1+j)%len(markers)],markevery=10)
    subgraph.legend()
    if scale == 'log':
        subgraph.set_yscale('log')

    subgraph.set_xlim(left=0,right=base_network.train_time_stamps[-1])
    subgraph.set_ylim(bottom = 0,top = 3.0)
    return subgraph

def plot_all_graphs(base_network, networks_to_compare = [], download = False, networks_type_str = 'Regular', scale = 'regular', figsize = (6,3.65),title_size=15,general_size=10.5,axis_size=14.5,title_height=0.96,fontstyle='normal',fontbold='normal'):

    if networks_type_str == 'Linear':
        title = 'Fully Connected, Linear Activation'
        figsize = (6,3.05)
        general_size = 9
    if networks_type_str == 'ReLU':
        title = 'Fully Connected, Rectified Linear Activation'
        figsize = (6, 3.05)
        general_size = 9
    if networks_type_str == 'conv_subsample':
        title = 'Convolutional, Adapted'
    if networks_type_str == 'conv_max_pool':
        title = 'Convolutional, Off-the-Shelf'

    figure, subgraphs = plt.subplots(1, 2, figsize=figsize) # For fully connected used figsize = (6,3.05)

    plt.rcParams.update({'legend.fontsize': general_size})

    figure.suptitle(title,fontstyle=fontstyle,fontweight=fontbold,fontsize=title_size,y=title_height)

    set_networks_loss_subgraph(base_network,subgraphs[0],networks_to_compare=networks_to_compare, download=download, networks_type_str=networks_type_str, scale=scale,axis_size=axis_size)
    set_networks_distance_subgraph(base_network,subgraphs[1],networks_to_compare=networks_to_compare , download=download, networks_type_str=networks_type_str, scale=scale,axis_size=axis_size)

    plt.tight_layout()

    if download == True:
        plt.savefig('Graph_'+networks_type_str.lower()+'.pdf',bbox_inches = "tight")

    plt.show()

def experiment(compare_ratios = [],batch_size_train = 1000,n_epochs = 10000,learning_rate = 0.01,deviation = 0.01,layers_size=[28*28,50,50,10],activation='Linear', download = False, scale = 'log'):

    #data
    train_loader , test_loader = data_load(batch_size_train)

    #initialize networks
    network = Network(train_loader,test_loader, label=activation+'_h='+str(learning_rate),deviation=deviation, layers_size=layers_size, activation=activation,batch_size_train=batch_size_train, init_mode='xavier')
    network = network.to('cuda')
    network.save_state()
    networks_compare = []
    for ratio in compare_ratios:
        cur_network = torch.load(activation+'_h='+str(learning_rate) + '_model.pth')
        cur_network = cur_network.to('cuda')
        cur_network.label = activation+'_h='+str(learning_rate/ratio)
        networks_compare.append(cur_network)

    #train networks
    network.init_train_params(n_epochs,learning_rate)
    network.train()
    for ratio,network_to_compare in zip(compare_ratios,networks_compare):
        network_to_compare.init_train_params(n_epochs * ratio, learning_rate/ratio,lr_ratio_from_base=ratio)
        network_to_compare.train()

    #plot
    #plot_all_graphs(network,networks_to_compare=networks_compare,networks_type_str=activation,download=download, scale=scale)

    #old plots
    #plot_networks_loss_graph(network,networks_to_compare=networks_compare,networks_type_str=activation,download=download)
    #plot_networks_loss_graph(network,networks_to_compare=networks_compare,networks_type_str=activation,download=download, scale='log')
    #plot_networks_distance_graph(network , networks_compare,networks_type_str=activation,download=download)
    #plot_networks_distance_graph(network , networks_compare,networks_type_str=activation,download=download, scale='log')

def continue_training(network_file_name = 'Linear_h=0.005_model.pth', compare_ratios = [],batch_size_train = 1000,n_epochs = 10000,learning_rate = 0.01,deviation = 0.01,layers_size=[28*28,50,50,10],activation='Linear', download = False):

    #data
    train_loader , test_loader = data_load(batch_size_train)
    network = torch.load(network_file_name)
    network = network.to('cuda')
    network.train()

def load_networks_file_paths(activation = 'Linear',stepsize = 0.001, ratios = [2,5,10,20]):
    networks_file_paths = []
    networks_file_paths.append(activation+'_h='+str(stepsize)+'_model.pth')
    for ratio in ratios:
        networks_file_paths.append(activation+'_h='+str(stepsize/ratio)+'_model.pth')
    return networks_file_paths

def plot_trained_networks_of_a_single_type(activation = 'Linear' , stepsize = 0.001, ratios = [2,5,10,20] , download = True, scale = 'regular'):
    network_file_paths = load_networks_file_paths(activation=activation,stepsize=stepsize,ratios=ratios)
    network = torch.load(network_file_paths[0])
    networks_compare = [torch.load(network_file_path) for network_file_path in network_file_paths[1:]]
    plot_all_graphs(network,networks_to_compare=networks_compare,networks_type_str=activation,download=download, scale=scale)

def plot_trained_networks_of_all_types(stepsize = 0.001, ratios = [2,5,10,20],scale = 'regular'):
    plot_trained_networks_of_a_single_type(activation='Linear',stepsize=stepsize,ratios=ratios,scale=scale)
    plot_trained_networks_of_a_single_type(activation='ReLU',stepsize=stepsize,ratios=ratios,scale=scale)
    plot_trained_networks_of_a_single_type(activation='conv_subsample',stepsize=stepsize,ratios=ratios,scale=scale)
    plot_trained_networks_of_a_single_type(activation='conv_max_pool',stepsize=stepsize,ratios=ratios,scale=scale)

def general_all_experiments(compare_ratios=[2,4],n_epochs=10000,learning_rate=0.001,batch_size_train=100):
    experiment(activation='Linear',compare_ratios=compare_ratios,n_epochs=n_epochs,learning_rate=learning_rate,download=True,batch_size_train=batch_size_train)
    experiment(activation='ReLU',compare_ratios=compare_ratios,n_epochs=n_epochs,learning_rate=learning_rate,download=True,batch_size_train=batch_size_train)
    experiment(activation='conv_subsample',compare_ratios=compare_ratios,n_epochs=n_epochs,learning_rate=learning_rate,download=True,batch_size_train=batch_size_train)
    experiment(activation='conv_max_pool',compare_ratios=compare_ratios,n_epochs=n_epochs,learning_rate=learning_rate,download=True,batch_size_train=batch_size_train)

def super_fast_experiments_to_check_graph_style():
    general_all_experiments(compare_ratios=[2,3],n_epochs=1000,learning_rate=0.01,batch_size_train=20)

def semi_short_experiments_1hour_run():
    general_all_experiments(compare_ratios=[2,4],n_epochs=10000,learning_rate=0.001,batch_size_train=100)

def full_scale_experiments():
    general_all_experiments(compare_ratios=[2,5,10,20],n_epochs=10000,learning_rate=0.001,batch_size_train=1000)

def main():

    #technical settings
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    full_scale_experiments()
    plot_trained_networks_of_all_types()

    #experiment(activation='conv_max_pool',compare_ratios=[2,5,10,20],n_epochs=10000,learning_rate=0.001,download=True,batch_size_train=150)
    #super_fast_experiments_to_check_graph_style()
    #semi_short_experiments_1hour_run()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()

# Tasks
    #TODO check weird conv_net where to smaller the step size, the closer it is to eta_0