from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader
from torch import no_grad
from torch.nn import Module
from torch.optim import Adam

from plotting import plot_comparison

def train(
    train_data, test_data,
    model, loss,
    epochs=10,
    lr=1e-2,
    batch=64,
) -> Tuple[Module, np.ndarray, np.ndarray]:
    train_dl = DataLoader(train_data, batch_size=batch, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=batch, shuffle=True)
    
    test_loss = evaluate(model, loss, test_dl)
    print(f"test_loss_before = {test_loss}\n")

    final_loss_train = []
    final_loss_test = [test_loss]
    
    optim = Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in train_dl:
            optim.zero_grad()
            y_hat = model(x)
            step_loss = loss(y_hat, y)
            losses.append(step_loss.item())
            step_loss.backward()
            optim.step()
            
        train_loss = np.mean(losses)
        test_loss = evaluate(model, loss, test_dl)
        final_loss_train.append(train_loss)
        final_loss_test.append(test_loss)
        print(f"train_loss[{epoch}] = {train_loss}")
        print(f"test_loss[{epoch}] = {test_loss}\n")
        
    final_loss_train.append(evaluate(model, loss, train_dl))
    return model, np.array(final_loss_train), np.array(final_loss_test)
            
def evaluate(model, loss, test_dl):
    model.eval()
    losses = []
    with no_grad():
        first = True
        for x, y in test_dl:
            y_hat = model(x)
            if first:
                plot_comparison(x, None, y_hat)
                first = False
            step_loss = loss(y_hat, y)
            losses.append(step_loss.item())
    return np.mean(losses)

