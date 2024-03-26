import numpy as np
import pandas as pd

from typing import *

import torch
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class HousePriceData(Dataset):

    def __init__(self, normalize: bool, train=True, val=False, add_noise: bool=False):
        self.train_data = pd.read_csv("./data/train_split.csv", index_col="Id")
        self.val_data = pd.read_csv("./data/val_split.csv", index_col="Id")
        self.test_data = pd.read_csv("./data/test.csv", index_col="Id")
        self.data = pd.concat([self.train_data, self.val_data, self.test_data], axis=0)
        self.labels = self.data["SalePrice"]
        # Some columns have too many missing values, use pd.isna() to count

        self.normalize = normalize
        self.train = train
        self.val = val
        self.add_noise = add_noise
        self.preprocess()

    def preprocess(self):
        self.data = self.data.drop(columns=["SalePrice", "PoolQC", "MiscFeature", "Alley",
                                            "Fence", "MasVnrType", "FireplaceQu"])
        self.data.fillna(method="bfill", inplace=True)
        # for col in self.data.columns:
        #     if self.data[col].dtype == "object":
        #         encoder = OrdinalEncoder()
        #         self.data[col] = encoder.fit_transform(self.data[col].values.reshape(-1, 1))
        self.data = pd.get_dummies(self.data)
        if self.normalize:
            self.data = (self.data - self.data.mean()) / self.data.std()


    def __len__(self):
        if self.train:
            return len(self.train_data)
        if self.val:
            return len(self.val_data)
        return len(self.test_data)

    def __getitem__(self, item):
        data = self.data[:len(self.train_data)]
        if self.val:
            data = self.data[len(self.train_data):len(self.train_data) + len(self.val_data)]
        elif not self.train:
            data = self.data[len(self.train_data)+len(self.val_data):]
        row = data.iloc[item]
        label = 0
        if self.train or self.val:
            label = self.labels.iloc[item] / 1_000
        if self.add_noise:
            row = row * (1 + 0.03*np.random.randn(len(row)))
            label = label * (1 + 0.01*np.random.randn())
        return data.index.values[item], torch.Tensor(row), torch.Tensor([label])



if __name__ == '__main__':
    house_data = HousePriceData(True, train=True, val=False, add_noise=True)
    dataloader = DataLoader(dataset=house_data, batch_size=64)
    print(len(house_data))
    for id, data, label in dataloader:
        print(data)