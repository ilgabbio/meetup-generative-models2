from typing import Optional
from math import floor, ceil, sqrt

import numpy as np
from torch import Tensor
from matplotlib import pyplot as plt


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
    if x is not None and len(x.shape) == 4:
        plot_images(x[:at_most,0,:,:], "Input")
    if y is not None and len(y.shape) == 4:
        plot_images(y[:at_most,0,:,:], "GT")
    if y_hat is not None and len(y_hat.shape) == 4:
        plot_images(y_hat[:at_most,0,:,:], "Output")


def plot_metrics(names, metrics):
    def plot_graph(names, metrics, pt):
        for values in metrics:
            pt.plot(values)
        pt.legend(names)
    
    is_loss = np.array([name.endswith("_loss") for name in names])
    
    if all(is_loss):
        plt.figure(figsize=(6,2))
        plot_graph(names, metrics, plt)
        plt.title("Metrics")
        plt.show();
        return 
    
    names = np.array(list(names))
    metrics = np.array(list(metrics))

    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    plot_graph(names[is_loss], metrics[is_loss,:], ax[0])
    ax[0].title.set_text("Metrics")
    plot_graph(names[~is_loss], metrics[~is_loss,:], ax[1])
    ax[1].title.set_text("Other")
    plt.show()


def plot_embedding_space(embeddings, labels, title):
    fig = plt.figure()
    plt.scatter(embeddings[:,0],embeddings[:,1],c=labels,marker='.')
    plt.title(title)
    plt.show()


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

