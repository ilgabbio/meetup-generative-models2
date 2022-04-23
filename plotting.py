from typing import Optional
from torch import Tensor
from matplotlib import pyplot as plt
from math import floor, ceil, sqrt


def plot_images(images: Tensor, title: Optional[str] = None):
    """Images received as an (N,H,W) tensor."""
    n = images.shape[0]
    plt.figure(figsize=(n,1))
    for i in range(n):
        ax = plt.subplot(1,n,i+1)
        if i==0:
            plt.title(title)
        image = images[i,:,:].numpy()
        plt.imshow(image, cmap='gray')
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
    plt.show()


def plot_comparison(x, y, y_hat, at_most = 16):
    if x is not None:
        plot_images(x[:at_most,0,:,:], "Input")
    if y is not None:
        plot_images(y[:at_most,0,:,:], "GT")
    if y_hat is not None:
        plot_images(y_hat[:at_most,0,:,:], "Output")


def plot_losses(train, test):
    plt.figure()
    plt.plot(train)
    plt.plot(test)
    plt.title("Training losses")
    plt.legend(["Train loss","Test loss"])
    plt.show();


def plot_byvar(data, plotter):
    n = data.shape[1]
    rows = floor(sqrt(n))
    cols = ceil(n / rows)
    fig, ax = plt.subplots(rows, cols, figsize=(2*cols,2*rows))
    for i in range(n):
        plotter(ax[i//cols,i%cols], data[:,i])
    plt.show()


def plot_byvar_hist(data, bins=20):
    plot_byvar(data, lambda ax, var: ax.hist([var],bins))

