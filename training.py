import os
import pickle
from typing import Tuple, Dict
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import Tensor, no_grad
from torch.nn import Module
from torch.optim import Adam

from plotting import plot_comparison, plot_metrics

def train(
    train_data, test_data,
    model, loss,
    epochs=10,
    lr=1e-2,
    batch=64,
    metrics = None
) -> Tuple[Module, Dict[str,np.ndarray]]:
    train_dl = DataLoader(train_data, batch_size=batch, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=batch, shuffle=True)
    
    if metrics is None:
        metrics = MetricsCollector()
    optim = Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            optim.zero_grad()
            y_hat = model(x)
            step_loss = metrics.extract_loss("train", loss(y_hat, y))
            step_loss.backward()
            optim.step()
            
        evaluate(model, loss, metrics, test_dl)
        metrics.epoch()
        metrics.print_last()
        
    return model, metrics

            
class MetricsCollector:
    def __init__(self, *other_names):
        self._other_names = other_names
        self._data = defaultdict(list)
        self._epoch_data = defaultdict(list)

    def extract_loss(self, phase: str, loss_value) -> Tensor:
        if isinstance(loss_value, tuple):
            other_values = loss_value[1:]
            loss_value = loss_value[0]
            for i, value in enumerate(other_values):
                name = self._other_names[i] if len(self._other_names) > i else f"other[{i}]"
                self.collect(f"{phase}_{name}", value.item())
        self.collect(f"{phase}_loss", loss_value.item())
        return loss_value

    def collect(self, name, value):
        self._epoch_data[name].append(value)

    def epoch(self):
        for name, values in self._epoch_data.items():
            self._data[name].append(np.mean(values))
        self._epoch_data.clear()

    @property
    def metrics(self) -> Dict[str,np.ndarray]:
        return {
            name: np.array(values)
            for name, values in self._data.items()
        }

    def print_last(self):
        for name, values in self._data.items():
            print(f"{name}[{len(values)}] = {values[-1]}")
        print()

    def plot(self):
        data = self.metrics
        plot_metrics(data.keys(), data.values())


def evaluate(model, loss, metrics, test_dl):
    model.eval()
    with no_grad():
        first = True
        for x, y in test_dl:
            y_hat = model(x)
            if first:
                images = y_hat
                if isinstance(images, tuple):
                    images = images[0]
                plot_comparison(x, None, images)
                first = False
            step_loss = metrics.extract_loss("test",loss(y_hat, y))


def embedding(encoder, data):
    with no_grad():
        return np.vstack([
            encoder(data[i:i+1,None,:,:]).numpy()
            for i in range(data.shape[0])
        ])


def generate_images(encoder, decoder, n = 16):
    with no_grad():
        encoded = encoder(torch.zeros(1,1,28,28))
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        num_feats = encoded.shape[1]
        z = torch.randn(n,num_feats)
        images = decoder(z)
        return images[:,0,:,:]


def save_model(path, model: Module, metrics: MetricsCollector):
    # Saving the model:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path + '.model.pickle')

    # Saving the metrics:
    with open(path + '.metrics.pickle', 'wb') as f:
        pickle.dump(metrics, f)


def load_model(path, model) -> Tuple[Module, MetricsCollector]:
    # Loading the model:
    model.load_state_dict(torch.load(path + '.model.pickle'))
    model.eval()

    # Loading the metrics:
    with open(path + '.metrics.pickle', 'rb') as f:
        metrics = pickle.load(f)

    # Return all:
    return model, metrics

