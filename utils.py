# -*- coding: utf-8 -*-

#@title imports & settings { form-width: "10%" }
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib import lines as plt_lines
from matplotlib import markers as plt_markers

#@title Data load { form-width: "10%" }
from model import Network


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
        plt.savefig('plots/'+'Graph_'+networks_type_str.lower()+'.png',bbox_inches = "tight")

    plt.show()

def experiment(compare_ratios = [],batch_size_train = 1000,n_epochs = 10000,learning_rate = 0.01,deviation = 0.01,layers_size=[28*28,50,50,10],activation='Linear', download = False, scale = 'log'):

    #data
    train_loader , test_loader = data_load(batch_size_train)

    #initialize networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = Network(train_loader, test_loader, label=activation + '_h=' + str(learning_rate), deviation=deviation, layers_size=layers_size, activation=activation, batch_size_train=batch_size_train, init_mode='xavier')
    network = network.to(device)
    network.save_state()
    networks_compare = []
    for ratio in compare_ratios:
        cur_network = torch.load(activation+'_h='+str(learning_rate) + '_model.pth')
        cur_network = cur_network.to(device)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
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

def general_all_experiments(compare_ratios=[2,5,10,20],n_epochs=10000,learning_rate=0.001,batch_size_train=1000):
    experiment(activation='Linear',compare_ratios=compare_ratios,n_epochs=n_epochs,learning_rate=learning_rate,download=True,batch_size_train=batch_size_train)
    experiment(activation='ReLU',compare_ratios=compare_ratios,n_epochs=n_epochs,learning_rate=learning_rate,download=True,batch_size_train=batch_size_train)
    experiment(activation='conv_subsample',compare_ratios=compare_ratios,n_epochs=n_epochs,learning_rate=learning_rate,download=True,batch_size_train=batch_size_train)
    experiment(activation='conv_max_pool',compare_ratios=compare_ratios,n_epochs=n_epochs,learning_rate=learning_rate,download=True,batch_size_train=batch_size_train)

def run_specific_experiment(experiment_type=None,compare_ratios=[2,5,10,20],n_epochs=10000,learning_rate=0.001,batch_size_train=1000):

    if experiment_type == 'fully_connected_linear':
        activation = 'Linear'
    elif experiment_type == 'fully_connected_relu':
        activation = 'ReLU'
    elif experiment_type == 'conv_subsample':
        activation = 'conv_subsample'
    elif experiment_type == 'conv_maxpool':
        activation = 'conv_max_pool'
    else:
        print("no such experiment :\'",experiment_type,"\'")
        return

    experiment(activation=activation, compare_ratios=compare_ratios, n_epochs=n_epochs, learning_rate=learning_rate,
               download=True, batch_size_train=batch_size_train)
    plot_trained_networks_of_a_single_type(activation=activation,stepsize=learning_rate,ratios=compare_ratios)

def super_fast_experiments_to_check_graph_style():
    general_all_experiments(compare_ratios=[2,3],n_epochs=1000,learning_rate=0.01,batch_size_train=20)

def semi_short_experiments_1hour_run():
    general_all_experiments(compare_ratios=[2,4],n_epochs=10000,learning_rate=0.001,batch_size_train=100)

def full_scale_experiments():
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    general_all_experiments(compare_ratios=[2,5,10,20],n_epochs=10000,learning_rate=0.001,batch_size_train=1000)