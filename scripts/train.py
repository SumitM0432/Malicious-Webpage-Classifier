import os
import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import warnings
import model_dispatcher
from config_loader import config
import src.dataset as dataset
from src.cross_val import create_folds
from src.eval_metrics import metric_scores
from preprocessing import data_preprocessing

warnings.filterwarnings('ignore')

def create_dataloader(df, batch_size = int(config['BATCH_SIZE'])):
    cls = dataset.MaliciousBenignData(df)
    return DataLoader(
        cls,
        batch_size = batch_size,
        num_workers = 0)

def binary_acc(predictions, y_test):
    y_pred = torch.round(torch.sigmoid(predictions))
    correct = (y_pred == y_test).sum().float()
    acc = torch.round((correct/y_test.shape[0])*100)
    return acc

def train_model(model, device, data_loader, optimizer, criterian, epochs_n = int(config['EPOCHS'])+1):
    # Putting the model in training mode
    model.train()

    for epoch in range(1, epochs_n):
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

    print ("INGESTING THE DATA")
    # Importing the dataset
    df = pd.read_csv(config['DATASET_PATH']['TRAINING_FILE'])
    df.drop(columns = "Unnamed: 0", inplace = True)

    print ("RUNNING DATA PREPROCESSING")
    # Preprocessing
    df = data_preprocessing(df)

    print ("CREATING FOLDS")
    # Cross Validation
    df = create_folds(df)

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

        print ("TRAINING THE MODEL...")
        clf = model_dispatcher.models[models]
        clf.fit(X_train, y_train)

        preds = clf.predict(X_valid)

        print ("EVALUATING THE PERFORMANCE")
        metric_scores(y_valid, preds)

        print ("SAVING THE MODEL :)")
        joblib.dump(
            clf,
            os.path.join(config['OUTPUTS']['MODEL_OUTPUT'], f"{models}.bin")
        )

    elif models == 'dnn':
        df_train_loader = create_dataloader(df_train, batch_size = int(config['BATCH_SIZE']))
        df_valid_loader = create_dataloader(df_valid, batch_size = 1)

        model = model_dispatcher.dnn()
        model.to(config['DEVICE'])

        criterian = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr = float(config['LEARNING_RATE']))

        print (model)

        print ("TRAINING THE MODEL")
        train_model(model, config['DEVICE'], df_train_loader, optimizer, criterian)

        y_valid, preds = eval_model(model, config['DEVICE'], df_valid_loader)

        print ("TESTING THE MODEL")
        metric_scores(y_valid, preds)

        print ("SAVING THE MODEL :)")
        torch.save(model.state_dict(), config['OUTPUTS']['MODEL_OUTPUT'] + '/dnn.pth')

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