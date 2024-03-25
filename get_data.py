import numpy as np
import pandas as pd

from typing import *

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class HousePriceData(Dataset):

    def __init__(self, normalize: bool, train=True):
        self.train_data = pd.read_csv("./data/train.csv", index_col="Id")
        self.test_data = pd.read_csv("./data/test.csv", index_col="Id")
        self.data = pd.concat([self.train_data, self.test_data], axis=0)
        self.labels = self.data["SalePrice"]
        self.data = self.data.drop(columns=["SalePrice"])
        self.data = pd.get_dummies(self.data)
        self.data.fillna(method="bfill", inplace=True)
        self.normalize = normalize
        self.train = train
        self.preprocess()

    def preprocess(self):
        if self.normalize:
            self.data = (self.data - self.data.mean()) / self.data.std()

    def __len__(self):
        if self.train:
            return len(self.train_data)
        return len(self.test_data)

    def __getitem__(self, item):
        data = self.data[:len(self.train_data)]
        if not self.train:
            data = self.data[len(self.train_data):]
        row = data.iloc[item]
        label = 0
        if self.train:
            label = self.labels.iloc[item] / 100_000
        return data.index.values[item], torch.Tensor(row), torch.Tensor([label])



if __name__ == '__main__':
    house_data = HousePriceData(True, train=True)
    dataloader = DataLoader(dataset=house_data, batch_size=64)
    print(len(house_data))
    for id, data, label in dataloader:
        print(label)