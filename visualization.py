import matplotlib.pyplot as plt


def plot_result(encoder, conv_vae, x_test, cwd):
    x_pred = conv_vae.predict(x_test)
    n = 20
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