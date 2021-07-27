import torch
import argparse
import utils as main_file

def main():
    #technical settings
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="fully_connected_linear", help="type of experiment to run, options are: \n1. fully_connected_linear \n2. fully_connected_relu \n3. conv_subsample \n4. conv_maxpool ")
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="size of the learning rate")
    args = parser.parse_args()
    main_file.run_specific_experiment(experiment_type=args.experiment,n_epochs=args.epochs,learning_rate=args.learning_rate)

if __name__ == "__main__":
    main()