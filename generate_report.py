def generate_report(cwd, encoder, decoder, conv_vae, filters, strides, num_layers, kernel_size, fcl, loss, opt, learning_rate, latent_dim, epochs, batch_size, validation_split, x_train, x_test, early_stopping,\
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
        f.write(
            """
            batch_size = {}
            kernel_size = {}
            filters = {}
            strides = {}
            nb units in dense layer = {}
            num_layers = {}
            latent_dim = {}
            epochs = {}
            validation_split = {}
            loss function = {}
            opt = {}
            learning_rate = {}            
            early_stopping = {}
            annealing = {}
            input_shape = {}
            train data shape = {}
            test data shape = {}
            """.format(
                batch_size,
                kernel_size,
                filters,
                strides,
                fcl,
                num_layers,
                latent_dim,
                epochs,
                validation_split, 
                loss,
                opt,
                learning_rate,
                early_stopping,
                annealing,
                input_shape,
                x_train.shape,
                x_test.shape
            )
        )
        if early_stopping:
            f.write('call back monitor: {}\n'.format(monitor))
            f.write('call back min delta: {}\n'.format(min_delta))
            f.write('call back patience: {}\n'.format(patience))
        if annealing:
            f.write('starting epoch for annealing: {}\n'.format(klstart))
            f.write('increasing step for annealing: {}\n'.format(kl_annealtime))

    conv_vae.save(cwd+'/conv_vae')     
