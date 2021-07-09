import tensorflow as tf
from tensorflow.keras import losses
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from tensorflow import keras
import os
from pathlib import Path
from tensorflow.keras.callbacks import Callback
from datetime import datetime
from create_model import create_model
from tensorflow.keras import layers, Input, Model
from tensorflow.keras import backend as K


def load_data(path):
    data = []
    for file in os.listdir(path):
        temp_data = np.load(os.path.join(path, file))
        data.append(temp_data)

    data = np.array(data)
    data = data.reshape(-1,75,80)
    return data

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


def sampling(args):
    z_mean, z_log_sigma, latent_dim = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon



if __name__ == '__main__':
    
    print('start loading data')    
    # data = load_data('../SeminarAF')
    data = load_data('../sample_spectrogram')

    print('data loaded')
    data_len = len(data)
    test_size = round(data_len*0.25)
    index = np.random.permutation(data_len)
    normalized_data = data/np.max(np.abs(data), axis=0)
    x_test = normalized_data[index[:test_size]]
    x_train = normalized_data[index[test_size:]]
    x_train = np.reshape(x_train, (len(x_train), 75, 80, 1))
    x_test = np.reshape(x_test, (len(x_test), 75, 80, 1))

    path = 'spectrogram_models/'
    now = datetime.now()
    cwd = path+now.strftime("%Y-%m-%d_%H-%M-%S")
    Path(cwd).mkdir()
    opt = 'adam'
    epochs = 40
    batch_size = 128
    validation_split = 0.1
    early_stopping = True
    monitor = 'val_loss'
    min_delta=1e-4
    mode='auto'
    patience=3
    annealing = True
    klstart = 2
    kl_annealtime = 3
    latent_dim = 30


    print("start creating the model")

    # conv_vae, encoder, decoder, z_mean, z_log_sigma, encoder_inp = create_model(latent_dim)
    conv_vae, encoder, decoder, z_mean, z_log_sigma, encoder_inp = create_model(latent_dim)

    class AnnealingCallback(Callback):
        def __init__(self, weight):
            self.weight = weight
        def on_epoch_end (self, epoch, logs={}):
            if epoch > klstart :
                new_weight = min(K.get_value(self.weight) + (1./ kl_annealtime), 2)
                K.set_value(self.weight, new_weight)
            print ("Current KL Weight is " + str(K.get_value(self.weight)))

    weight = K.variable(0.)

    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(losses.mean_squared_error(encoder_inp, decoder(encoder(encoder_inp)[2])),axis=(1,2)))
    kl_loss = 1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma)
    kl_loss = -0.5*tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    
    callback_early_stopping = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta,
                                        patience=patience, verbose=0, mode=mode,
                                        baseline=None, restore_best_weights=True)
    if not early_stopping:
        callback_early_stopping = None

    print("start training the model")

    if not annealing:
        conv_vae_loss = K.mean(kl_loss + reconstruction_loss)
        conv_vae.add_loss(conv_vae_loss)
        conv_vae.compile(optimizer=opt)
        conv_vae.fit(x_train, x_train, validation_split=validation_split, epochs=epochs, callbacks=[callback_early_stopping], batch_size=batch_size)
    else:
        conv_vae.add_loss(reconstruction_loss)
        conv_vae.add_loss(kl_loss)
        annealing_callback = AnnealingCallback(weight)
        conv_vae.compile(optimizer=opt)
        conv_vae.fit(x_train, x_train, validation_split=validation_split, epochs=epochs, callbacks=[callback_early_stopping, annealing_callback], batch_size=batch_size)
    
    
    print("write report")
    generate_report(cwd, encoder, decoder, conv_vae)


    x_pred = conv_vae.predict(x_test)
    n = 10
    plt.figure(figsize=(35,5))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i][:,:,0])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i+n+1)
        plt.imshow(x_pred[i][:,:,0])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(cwd+'/prediction')

    mu, log_sigma, _ = encoder.predict(x_test)
    plt.figure(figsize=(10,10))
    plt.scatter(mu[:, 0], mu[:, 1], cmap='plasma')
    plt.colorbar()
    plt.savefig(cwd+'/mu scatter plot')