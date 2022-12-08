from torch import nn
import torch

def train(model, loader, optimizer, clip, loss_fn):
    model.train()
    epoch_loss = 0
    for X, Y in loader:
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output.to(torch.float32), Y.to(torch.float32))
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate(model, loader, loss_fn):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for X, Y in loader:
            output = model(X)
            loss = loss_fn(output.to(torch.float32), Y.to(torch.float32))
            epoch_loss += loss.item()
    return epoch_loss / len(loader)
