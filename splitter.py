import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    train_data = pd.read_csv("./data/train.csv", index_col="Id")
    train, validation = train_test_split(train_data, test_size=0.15, random_state=42)
    train.to_csv("./data/train_split.csv")
    validation.to_csv("./data/val_split.csv")