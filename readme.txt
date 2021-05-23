The code requires pytorch with CUDA.
To reconstruct the experiments just run main.

The function call 'full_scale_experiments()' trains all the models, and saves a copy of the networks when finished training (using pickle).

The function call 'plot_trained_networks_of_all_types()' loads the pickled network models and creates the plots.