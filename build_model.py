from tensorflow.keras import losses
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from tensorflow.keras import backend as K



class AnnealingCallback(Callback):
        def __init__(self, weight, klstart, kl_annealtime):
            self.weight = weight
            self.klstart = klstart
            self.kl_annealtime = kl_annealtime
        def on_epoch_end (self, epoch, logs={}):
            if epoch > self.klstart :
                new_weight = min(K.get_value(self.weight) + (1./ self.kl_annealtime), 2)
                K.set_value(self.weight, new_weight)
            print ("Current KL Weight is " + str(K.get_value(self.weight)))



def build_model(encoder_inp, encoder, decoder, conv_vae,z_mean, z_log_sigma, monitor, min_delta, patience, klstart, kl_annealtime, \
    validation_split, epochs, batch_size, opt, learning_rate, early_stopping, annealing, x_train, cwd):

    weight = K.variable(0.)
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(losses.mean_squared_error(encoder_inp, decoder(encoder(encoder_inp)[2])),axis=(1,2)))
    kl_loss = 1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma)
    kl_loss = -0.5*tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    
    callback_early_stopping = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta,
                                        patience=patience, verbose=0, mode='auto',
                                        baseline=None, restore_best_weights=True)
    if not early_stopping:
        callback_early_stopping = None

    print("start training the model")
    opt = keras.optimizers.Adam(learning_rate)
    if not annealing:
        conv_vae_loss = K.mean(kl_loss + reconstruction_loss)
        conv_vae.add_loss(conv_vae_loss)
        conv_vae.compile(optimizer=opt)
        history = conv_vae.fit(x_train, x_train, validation_split=validation_split, epochs=epochs, callbacks=[callback_early_stopping], batch_size=batch_size)
    else:
        conv_vae.add_loss(reconstruction_loss)
        conv_vae.add_loss(kl_loss)
        annealing_callback = AnnealingCallback(weight, klstart, kl_annealtime)
        conv_vae.compile(optimizer=opt)
        history = conv_vae.fit(x_train, x_train, validation_split=validation_split, epochs=epochs, callbacks=[callback_early_stopping, annealing_callback], batch_size=batch_size)
    
    conv_vae.save(cwd + '/vae_cnn_spec.h5')
    return conv_vae, history