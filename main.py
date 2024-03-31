import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from get_data import *
from model import *
from tqdm import tqdm
from sklearn.ensemble import *
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR


def train(max_epochs, batch_size, normalize):
    dataset = HousePriceData(normalize=normalize, train=True, val=False, add_noise=False)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    dataset = HousePriceData(normalize, False, True)
    val_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    # model = ANN(input_dim=266, hidden_layers=[64, 32, 16], output_dim=1)
    model = ResidualANN(input_dim=266, hidden_layer=256, number_of_hidden=4, output_dim=1)
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

        bar.set_description(
            f"EPOCH: {epoch} | LAST TRAIN LOSS: {np.mean(all_train_loss):.3} | LAST VAL LOSS: {np.mean(all_val_loss):.3}")
        train_loss.append(np.mean(all_train_loss))
        val_loss.append(np.mean(all_val_loss))
    plt.plot(train_loss, label='Train loss')
    plt.plot(val_loss, label='Validation loss')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()
    return model


def train_sklearn(normalize):
    # By using dataset length we load everything at one
    dataset = HousePriceData(normalize=normalize, train=True, val=False, add_noise=False)
    dataloader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=True)
    dataset = HousePriceData(normalize, False, True)
    val_dataloader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=True)
    model = GradientBoostingRegressor()
    for ids, data, labels in dataloader:
        labels = labels.view(len(labels))
        model.fit(data, labels)
    for ids, data, labels in val_dataloader:
        labels = labels.view(len(labels))
        score = model.score(data, labels)
    return model, score


def hyperparameter_tuning(model, parameters, normalize):
    grid = GridSearchCV(estimator=model, param_grid=parameters, n_jobs=-1, verbose=2)

    dataset = HousePriceData(normalize=normalize, train=True, val=False, add_noise=False)
    dataloader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=True)
    for ids, data, labels in dataloader:
        labels = labels.view(len(labels))
        grid.fit(data, labels)
    print(f"Best params: {grid.best_params_}")
    dataset = HousePriceData(normalize, False, True)
    val_dataloader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=True)
    for ids, data, labels in val_dataloader:
        labels = labels.view(len(labels))
        score = grid.best_estimator_.score(data, labels)
    return grid.best_estimator_, score


def make_submission_torch(model, normalize):
    dataset = HousePriceData(normalize, False, False)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    result = pd.DataFrame(columns=["SalePrice"])
    model.eval()
    for id, data, _ in dataloader:
        pred = model(data)
        result.loc[id.item()] = 10_000 * pred.item()
    result.to_csv("submission.csv", index_label="Id")


def make_submission_sklearn(model, normalize, pca=False, pca_var=0.99):
    dataset_train = HousePriceData(normalize, train=True, val=False, pca=pca, pca_var=pca_var)
    dataset_val = HousePriceData(normalize, train=False, val=True, pca=pca, pca_var=pca_var)
    dataset_comb = torch.utils.data.ConcatDataset([dataset_train, dataset_val])
    dataloader = DataLoader(dataset=dataset_comb, batch_size=len(dataset_comb), shuffle=True)
    for ids, data, labels in dataloader:
        labels = labels.view(len(labels))
        model.fit(data, labels)

    dataset = HousePriceData(normalize, False, False, pca=pca, pca_var=pca_var)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    result = pd.DataFrame(columns=["SalePrice"])
    for id, data, _ in dataloader:
        pred = model.predict(data)
        result.loc[id.item()] = 10_000 * pred.item()
    result.to_csv(f"submission_{type(model).__name__}.csv", index_label="Id")


def lasso_regression(folds=10, pca=True, pca_var=0.99, normalize=False):
    dataset_train = HousePriceData(normalize, train=True, val=False, pca=pca, pca_var=pca_var)
    dataset_val = HousePriceData(normalize, train=False, val=True, pca=pca, pca_var=pca_var)
    dataset_comb = torch.utils.data.ConcatDataset([dataset_train, dataset_val])

    model = LassoCV(cv=folds)

    dataloader = DataLoader(dataset=dataset_comb, batch_size=len(dataset_comb), shuffle=True)
    for ids, data, labels in dataloader:
        labels = labels.view(len(labels))
        model.fit(data, labels)

    return model


def kernel_regression(folds=10, pca=True, pca_var=0.99, normalize=False):
    grid = GridSearchCV(KernelRidge(kernel="rbf", gamma=0.1),
                        param_grid={"alpha": [1e-3, 1e-2, 1e-1, 1., 5., 10.],
                                    "gamma": np.logspace(-2, 2, 5)},
                        refit=True, cv=folds, n_jobs=-1, verbose=1)

    dataset_train = HousePriceData(normalize, train=True, val=False, pca=pca, pca_var=pca_var)
    dataset_val = HousePriceData(normalize, train=False, val=True, pca=pca, pca_var=pca_var)
    dataset_comb = torch.utils.data.ConcatDataset([dataset_train, dataset_val])

    dataloader = DataLoader(dataset=dataset_comb, batch_size=len(dataset_comb), shuffle=True)
    for ids, data, labels in dataloader:
        labels = labels.view(len(labels))
        grid.fit(data, labels)

    return grid.best_estimator_


if __name__ == '__main__':
    normalize = True
    pca = True
    # pca_var = 0.99    # 0 < x < 1. float sets minimum variance captured by vars
    pca_var = 'mle'     # mle setting uses some MLE method to guess best dimension to use (~227 vars)

    # model = lasso_regression(pca=pca, pca_var=pca_var, normalize=normalize)
    # model = kernel_regression(pca=pca, pca_var=pca_var, normalize=normalize)
    # make_submission_sklearn(model, normalize, pca, pca_var)

    # model = train(50, 256, normalize)
    # make_submission_torch(model, normalize)
    # model, score = train_sklearn(normalize)

    # model = GradientBoostingRegressor(random_state=42)
    # # parameters = {
    # #     "loss": ["squared_error", "absolute_error", "huber"],
    # #     "learning_rate": [1e-2, 1e-1, 0.2, 0.3],
    # #     "n_estimators": [256, 512, 600, 700]
    # # }
    # # parameters = {
    # #     "loss": ["squared_error"],
    # #     "learning_rate": [0.15, 0.2, 0.25],
    # #     "n_estimators": [550, 600, 650]
    # # }
    # parameters = {
    #     "loss": ["squared_error"],
    #     "learning_rate": [0.12, 0.15, 0.17],
    #     "n_estimators": [630, 650, 670]
    # }
    #
    # model, score = hyperparameter_tuning(model, parameters, normalize)
    # print(f"Gradient Boosting score: {score}")
    #
    # make_submission_sklearn(model, normalize)
    #
    # model = AdaBoostRegressor(random_state=42)
    # # parameters = {
    # #     "loss": ["linear", "square"],
    # #     "learning_rate": [1e-2, 1e-1, 1],
    # #     "n_estimators": [128, 256, 512, 600, 700]
    # # }
    # # parameters = {
    # #     "loss": ["linear"],
    # #     "learning_rate": [0.05, 1e-1, 0.15],
    # #     "n_estimators": [200, 256, 300]
    # # }
    # parameters = {
    #     "loss": ["linear"],
    #     "learning_rate": [0.12, 0.15, 0.17],
    #     "n_estimators": [170, 200, 230]
    # }
    #
    # model, score = hyperparameter_tuning(model, parameters, normalize)
    # print(f"AdaBoost score: {score}")
    #
    # make_submission_sklearn(model, normalize)
    #
    # model = RandomForestRegressor(n_jobs=-1, random_state=42)
    # parameters = {
    #     "min_samples_leaf": [2, 5, 7],
    #     "max_depth": [17],
    #     "n_estimators": [40, 50, 60]
    # }
    #
    # model, score = hyperparameter_tuning(model, parameters, normalize)
    # print(f"Random Forest score: {score}")

    model = SVR(max_iter=20_000)
    parameters = {
        "kernel": ["rbf"],
        "degree": [1],
        "C": [45, 46, 47, 48, 49, 50]
    }

    model, score = hyperparameter_tuning(model, parameters, normalize)
    print(f"SVR score: {score}")

    make_submission_sklearn(model, normalize)
