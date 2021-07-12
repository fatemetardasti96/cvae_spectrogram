from pathlib import Path
from datetime import datetime
import argparse

from numpy import random

from load_data import load_data
from visualization import plot_result
from build_model import build_model
from apply_reshaping import apply_reshaping
from create_model import create_model_flexible, create_model_rigid
from generate_report import generate_report


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="parameters to set for VAE model")
    parser.add_argument("--data-path-name", type=str, default="../SeminarAF",
                        help="where to load the data")
    parser.add_argument("--latent-dim", type=int, default=10,
                        help="latent dimension size")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="nb batch size")
    parser.add_argument("--nb-epochs", type=int, default=20,
                        help="nb epochs")
    parser.add_argument("--kernel-size", type=int, default=3,
                        help="kernel size")
    parser.add_argument("--filter-size", type=int, default=16,
                        help="initial filter size")
    parser.add_argument("--strides", type=int, default=2,
                        help="stride size")
    parser.add_argument("--nb-layer", type=int, default=3,
                        help="number of layers in VAE model")
    parser.add_argument("--validation-split", type=float, default=0.1,
                        help="validation split in fitting step")
    parser.add_argument("--opt", type=str, default="adam",
                        help="optimizer to compile the model")
    parser.add_argument("--monitor", type=str, default="val_loss",
                        help="monitor parameter for early stopping val_loss or loss")                        
    parser.add_argument("--min-delta", type=float, default=0.0001,
                        help="min delta in early stopping")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--patience", type=int, default=3,
                        help="patience in raly stopping")
    parser.add_argument("--early-stopping", type=bool, default=True,
                        help="stop the model if monitor value grows")
    parser.add_argument("--annealing", type=bool, default=True,
                        help="apply kl loss annealing (0/1)")
    parser.add_argument("--klstart", type=int, default=2,
                        help="start annealing after klstart epochs")
    parser.add_argument("--kl-annealtime", type=int, default=2,
                        help="growing weight for kl loss")
    parser.add_argument("--input-shape", type=tuple_type, default=(75, 80),
                        help="if different from default then should apply reshaping")
    parser.add_argument("--random", type=bool, default=True,
                        help="generate random or fixed train and test data (0/1)")


    args = parser.parse_args()
    
    data_path_name = args.data_path_name
    batch_size = args.batch_size
    kernel_size = args.kernel_size
    filters = args.filter_size
    strides = args.strides
    num_layers = args.nb_layer
    latent_dim = args.latent_dim
    epochs = args.nb_epochs
    validation_split = args.validation_split
    opt = args.opt
    learning_rate = args.learning_rate
    monitor = args.monitor
    min_delta = args.min_delta
    patience = args.patience
    early_stopping = args.early_stopping
    annealing = args.annealing
    klstart = args.klstart
    kl_annealtime = args.kl_annealtime
    input_shape = args.input_shape
    random = args.random

    
    print('start loading data')    
    x_train, x_test = load_data(data_path_name, random=True)

    if input_shape != (75, 80):
        x_train, x_test = apply_reshaping(x_train, x_test, input_shape)

    print('data loaded')


    path = 'spectrogram_models/'
    now = datetime.now()
    cwd = path+now.strftime("%Y-%m-%d_%H-%M-%S")
    Path(cwd).mkdir()


    print("start creating the model")

    # conv_vae, encoder, decoder, z_mean, z_log_sigma, encoder_inp = create_model(latent_dim)
    input_shape = (input_shape[0], input_shape[1], 1)
    conv_vae, encoder, decoder, z_mean, z_log_sigma, encoder_inp = create_model_rigid(input_shape, filters, kernel_size,\
         strides, latent_dim, num_layers)

    build_model(encoder_inp, encoder, decoder, conv_vae,z_mean, z_log_sigma, monitor, min_delta, patience, klstart, kl_annealtime, \
    validation_split, epochs, batch_size, opt, learning_rate, early_stopping, annealing, x_train, cwd)
    
    
    print("write report")
    generate_report(cwd, encoder, decoder, conv_vae, filters, strides, num_layers, kernel_size, opt, learning_rate, latent_dim, epochs, batch_size, validation_split, x_train, x_test, early_stopping,\
    monitor, min_delta, patience, annealing, klstart, kl_annealtime, input_shape)

    print("plot results")
    plot_result(encoder, conv_vae, x_test, cwd)
    