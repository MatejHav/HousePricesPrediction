import torch
import pandas as pd
import numpy as np

from get_data import *
from model import *
from tqdm import tqdm


def train(max_epochs, batch_size):
    dataset = HousePriceData(True, True)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    model = ANN(input_dim=288, hidden_layers=[256, 128, 64, 32, 16, 8, 4, 2], output_dim=1)
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = nn.MSELoss(reduction='mean')

    bar = tqdm(range(max_epochs))
    bar.set_description("EPOCH: 0 | LAST TRAIN LOSS: - | ")
    for epoch in bar:
        # Training
        for _, data, labels in dataloader:
            predictions = model(data)
            loss = loss_function(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Get training loss
        all_loss = []
        for _, data, labels in dataloader:
            with torch.no_grad():
                predictions = model(data)
                loss = loss_function(predictions, labels)
                all_loss.append(loss.item())
        bar.set_description(f"EPOCH: {epoch} | LAST TRAIN LOSS: {np.mean(all_loss):.3} | ")
    return model


def make_submission(model):
    dataset = HousePriceData(True, False)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    result = pd.DataFrame(columns=["SalePrice"])
    for id, data, _ in dataloader:
        pred = model(data)
        result.loc[id.item()] = 100_000 * pred.item()
    result.to_csv("submission.csv", index_label="Id")


if __name__ == '__main__':
    model = train(50, 146)
    make_submission(model)
