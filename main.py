import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from get_data import *
from model import *
from tqdm import tqdm


def train(max_epochs, batch_size, normalize):
    dataset = HousePriceData(normalize=normalize, train=True, val=False, add_noise=False)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    dataset = HousePriceData(normalize, False, True)
    val_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    model = ANN(input_dim=266, hidden_layers=[64, 32, 16], output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3)
    loss_function = nn.MSELoss()

    bar = tqdm(range(max_epochs))
    bar.set_description("EPOCH: 0 | LAST TRAIN LOSS: - | ")
    train_loss = []
    val_loss = []
    for epoch in bar:
        # Training
        model.train()
        for _, data, labels in dataloader:
            predictions = model(data)
            loss = loss_function(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        # Get training loss
        all_train_loss = []
        for _, data, labels in dataloader:
            with torch.no_grad():
                predictions = model(data)
                loss = loss_function(predictions, labels)
                all_train_loss.append(loss.item())

        # Get validation loss
        all_val_loss = []
        for _, data, labels in val_dataloader:
            with torch.no_grad():
                predictions = model(data)
                loss = loss_function(predictions, labels)
                all_val_loss.append(loss.item())

        bar.set_description(f"EPOCH: {epoch} | LAST TRAIN LOSS: {np.mean(all_train_loss):.3} | LAST VAL LOSS: {np.mean(all_val_loss):.3}")
        train_loss.append(np.mean(all_train_loss))
        val_loss.append(np.mean(all_val_loss))
    plt.plot(train_loss, label='Train loss')
    plt.plot(val_loss, label='Validation loss')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()
    return model


def make_submission(model, normalize):
    dataset = HousePriceData(normalize, False, False)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    result = pd.DataFrame(columns=["SalePrice"])
    model.eval()
    for id, data, _ in dataloader:
        pred = model(data)
        result.loc[id.item()] = 1_000 * pred.item()
    result.to_csv("submission.csv", index_label="Id")


if __name__ == '__main__':
    normalize = False
    model = train(50, 256, normalize)
    make_submission(model, normalize)
