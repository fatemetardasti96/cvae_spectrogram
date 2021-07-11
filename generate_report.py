def generate_report(cwd, encoder, decoder, conv_vae, opt, learning_rate, latent_dim, epochs, batch_size, validation_split, x_train, x_test, early_stopping,\
    monitor, min_delta, patience, annealing, klstart, kl_annealtime, input_shape):
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
        f.write('learning rate: {}\n'.format(learning_rate))
        f.write('latent dim: {}\n'.format(latent_dim))
        f.write('nb epochs: {}\n'.format(epochs))
        f.write('batch size: {}\n'.format(batch_size))
        f.write('validation split: {}\n'.format(validation_split))
        f.write('train data shape: {}\n'.format(x_train.shape))
        f.write('test data shape: {}\n'.format(x_test.shape))
        f.write('resize input data: {}\n'.format(input_shape))
        f.write('call back early stopping: {}\n'.format(early_stopping))
        if early_stopping:
            f.write('call back monitor: {}\n'.format(monitor))
            f.write('call back min delta: {}\n'.format(min_delta))
            f.write('call back patience: {}\n'.format(patience))
        if annealing:
            f.write('starting epoch for annealing: {}\n'.format(klstart))
            f.write('increasing step for annealing: {}\n'.format(kl_annealtime))

    conv_vae.save(cwd+'/conv_vae')     
