from pathlib import Path
from datetime import datetime
from create_model import create_model
import argparse

from load_data import load_data
from visualization import plot_result



def generate_report(cwd, encoder, decoder, conv_vae):
    encoder.save(cwd+'/encoder')
    with open(cwd+'/encoder_summary.txt', 'w') as f:
        encoder.summary(print_fn=lambda x: f.write(x + '\n'))

    decoder.save(cwd+'/decoder')
    with open(cwd+'/decoder_summary.txt', 'w') as f:
        decoder.summary(print_fn=lambda x: f.write(x + '\n'))

    with open(cwd+'/vae_summary.txt', 'w') as f:
        conv_vae.summary(print_fn=lambda x: f.write(x + '\n'))

    with open(cwd+'/parameter_summary.txt', 'w') as f:
        f.write('optimizer: {}\n'.format(opt))
        f.write('latent dim: {}\n'.format(latent_dim))
        f.write('nb epochs: {}\n'.format(epochs))
        f.write('batch size: {}\n'.format(batch_size))
        f.write('validation split: {}\n'.format(validation_split))
        f.write('train data shape: {}\n'.format(x_train.shape))
        f.write('test data shape: {}\n'.format(x_test.shape))
        f.write('min value of data: {}\n'.format(min(data.flatten())))
        f.write('max value of data: {}\n'.format(max(data.flatten())))
        f.write('call back early stopping: {}\n'.format(early_stopping))
        if early_stopping:
            f.write('call back monitor: {}\n'.format(monitor))
            f.write('call back min delta: {}\n'.format(min_delta))
            f.write('call back mode: {}\n'.format(mode))
            f.write('call back patience: {}\n'.format(patience))
        if annealing:
            f.write('starting epoch for annealing: {}\n'.format(klstart))
            f.write('increasing step for annealing: {}\n'.format(kl_annealtime))

    conv_vae.save(cwd+'/conv_vae')     



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
    parser.add_argument("--validation-split", type=float, default=0.1,
                        help="validation split in fitting step")
    parser.add_argument("--opt", type=str, default="adam",
                        help="optimizer to compile the model")
    parser.add_argument("--monitor", type=str, default="val_loss",
                        help="monitor parameter for early stopping val_loss or loss")                        
    parser.add_argument("--min-delta", type=float, default=0.0001,
                        help="min delta in early stopping")
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

    args = parser.parse_args()
    
    data_path_name = args.data_path_name
    batch_size = args.batch_size
    kernel_size = args.kernel_size
    filters = args.filter_size
    latent_dim = args.latent_dim
    epochs = args.nb_epochs
    validation_split = args.validation_split
    opt = args.opt
    monitor = args.monitor
    min_delta = args.min_delta
    patience = args.patience
    early_stopping = args.early_stopping
    annealing = args.annealing
    klstart = args.klstart
    kl_annealtime = args.kl_annealtime

    
    print('start loading data')    
    # x_train, x_test = load_data('../sample_spectrogram')
    x_train, x_test = load_data(data_path_name)

    print('data loaded')


    path = 'spectrogram_models/'
    now = datetime.now()
    cwd = path+now.strftime("%Y-%m-%d_%H-%M-%S")
    Path(cwd).mkdir()


    print("start creating the model")

    conv_vae, encoder, decoder, z_mean, z_log_sigma, encoder_inp = create_model(latent_dim)

    
    
    print("write report")
    generate_report(cwd, encoder, decoder, conv_vae)

    print("plot results")
    plot_result(encoder, conv_vae, x_test, cwd)
    