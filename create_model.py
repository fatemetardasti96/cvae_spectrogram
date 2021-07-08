from tensorflow.keras import layers, Input, Model
from tensorflow.keras import backend as K

def sampling(args):
    z_mean, z_log_sigma, latent_dim = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon



def create_model(latent_dim):    
    encoder_inp = Input(shape=(75, 80, 1))
    x = layers.Conv2D(64, (3, 2), strides=(1,1), activation='relu')(encoder_inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, (3, 2), activation="relu", strides=(2,2))(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", strides=(2,2))(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", strides=(2,2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_sigma = layers.Dense(latent_dim)(x)
    z = layers.Lambda(sampling)([z_mean, z_log_sigma, latent_dim])
    encoder = Model(encoder_inp, [z_mean, z_log_sigma, z], name='encoder')
    encoder.summary()

    latent_inputs = Input(shape=(latent_dim,))
    y = layers.Dense(100, activation='relu')(latent_inputs)
    y = layers.Dense(8*9*8, activation='relu')(y)
    y = layers.Reshape((8, 9, 8))(y)
    y = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='relu')(y)
    y = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='relu')(y)
    y = layers.Conv2DTranspose(128, (3, 2), strides=(2, 2), activation='relu')(y)
    y = layers.Conv2DTranspose(64, (3, 2), strides=(1, 1) ,activation='relu')(y)
    y = layers.Conv2DTranspose(1, (3, 2), activation='tanh')(y)
    decoded = layers.Lambda((lambda x: x*5))(y)
    decoder = Model(latent_inputs, decoded)
    decoder.summary()

    conv_vae = Model(encoder_inp, decoder(encoder(encoder_inp)[2]))

    return conv_vae, encoder, decoder, z_mean, z_log_sigma, encoder_inp