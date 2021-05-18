import config
import os
import joblib
import pandas as pd
import cross_val
from preprocessing import preprocessing
from torch.utils.data import DataLoader
import model_dispatcher
import dataset
import torch
import torch.nn as nn
import torch.optim as optim
import Metrics
import warnings
import argparse
warnings.filterwarnings('ignore')

def create_dataloader(df, batch_size):
    cls = dataset.MaliciousBenignData(df)
    return DataLoader(
        cls,
        batch_size = batch_size,
        num_workers = 0
    )

def binary_acc(predictions, y_test):
    y_pred = torch.round(torch.sigmoid(predictions))
    correct = (y_pred == y_test).sum().float()
    acc = torch.round((correct/y_test.shape[0])*100)
    return acc


def train_model(model, device, data_loader, optimizer, criterian):
    # Putting the model in training mode
    model.train()

    for epoch in range(1, config.EPOCHS+1):
        epoch_loss = 0
        epoch_acc = 0
        for X, y in data_loader:

            X = X.to(device)
            y_ = torch.tensor(y.unsqueeze(1), dtype = torch.float32)
            y = y_.to(device)

            # Zeroing the gradient
            optimizer.zero_grad()

            predictions = model(X.float())

            loss = criterian(predictions, y)
            acc = binary_acc(predictions, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        print (f"Epoch -- {epoch} | Loss : {epoch_loss/len(data_loader): .5f} | Accuracy : {epoch_acc/len(data_loader): .5f}")

def eval_model(model, device, data_loader):
    # Putting the model in evaluation mode
    model.eval()

    y_pred = []
    y_test_al = []

    with torch.no_grad():
        for X_test, y_test in data_loader:
            X_test = X_test.to(device)

            predictions = model(X_test.float())
            pred = torch.round(torch.sigmoid(predictions))

            y_test_al.append(y_test.tolist())
            y_pred.append(pred.tolist())

        # Changing the Predictions into list 
        y_test_al = [ele[0] for ele in y_test_al]
        y_pred = [int(ele[0][0]) for ele in y_pred]

        return (y_test_al, y_pred)

def run(folds, models):

    # Importing the dataset
    df = pd.read_csv(config.TRAINING_FILE)
    df.drop(columns = "Unnamed: 0", inplace = True)

    # Preprocessing
    df = preprocessing(df)

    # Cross Validation
    df = cross_val.create_folds(df)

    # training and validation set
    df_train = df[df.kfold != folds].reset_index(drop = True)
    df_train = df_train.drop(columns = ['kfold'])
    df_valid = df[df.kfold == folds].reset_index(drop = True)
    df_valid = df_valid.drop(columns = ['kfold'])

    if models in ['xg', 'lr', 'dt']:

        X_train = df_train.drop(columns = ['label'])
        y_train = df_train.label.values

        X_valid = df_valid.drop(columns = ['label'])
        y_valid = df_valid.label.values

        clf = model_dispatcher.models[models]
        clf.fit(X_train, y_train)

        preds = clf.predict(X_valid)

        joblib.dump(
            clf,
            os.path.join(config.MODEL_OUTPUT, f"{models}.bin")
        )

        Metrics.metric_scores(y_valid, preds)

    elif models == 'dnn':
        df_train_loader = create_dataloader(df_train, batch_size = config.BATCH_SIZE)
        df_valid_loader = create_dataloader(df_valid, batch_size = 1)

        model = model_dispatcher.dnn()
        model.to(config.DEVICE)

        criterian = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr = config.LEARNING_RATE)

        print (model)

        train_model(model, config.DEVICE, df_train_loader, optimizer, criterian)

        y_valid, preds = eval_model(model, config.DEVICE, df_valid_loader)

        torch.save(model.state_dict(), config.MODEL_OUTPUT + '/dnn.pth')

        Metrics.metric_scores(y_valid, preds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folds",
        type = int
    )

    parser.add_argument(
        "--model",
        type = str
    )

    args = parser.parse_args()

    run(
        folds = args.folds,
        models = args.model
    )
