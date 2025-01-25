import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__)) # Current Folder
project_root = os.path.abspath(os.path.join(current_dir, "..")) # Project Folder
sys.path.insert(0, project_root) # Setting the Project Folder as a priority to find modules

import torch
import src.dataset as dataset
import joblib
from preprocessing import data_preprocessing
import pandas as pd
import argparse
from src.Metrics import metric_scores
import model_dispatcher
from torch.utils.data import DataLoader
from src.config_loader import load_config
config = load_config()


def predict(path, model):

    df = pd.read_csv(path)

    df = data_preprocessing(df)

    if model in ['xg', 'lr', 'dt']:

        X_test = df.drop(columns = ['label'])
        y_test = df.label.values

        mod = joblib.load(config['paths']['MODEL_OUTPUT'] + str(model) + '.bin')
        
        predictions = mod.predict(X_test)

        metric_scores(y_test, predictions)
    
    elif model == 'dnn':
        model = model_dispatcher.dnn()
        model.to(config['DEVICE'])

        model.load_state_dict(torch.load(config['paths']['MODEL_OUTPUT'] + '/dnn.pth'))

        cls = dataset.MaliciousBenignData(df)
        
        df_test = DataLoader(
                            cls,
                            batch_size = 1,
                            num_workers = 0
                            )

        # Putting the model in evaluation mode
        model.eval()

        y_pred = []
        y_test_al = []

        with torch.no_grad():
            for X_test, y_test in df_test:
                X_test = X_test.to(config['DEVICE'])

                predictions = model(X_test.float())
                pred = torch.round(torch.sigmoid(predictions))

                y_test_al.append(y_test.tolist())
                y_pred.append(pred.tolist())

            # Changing the Predictions into list 
            y_test_al = [ele[0] for ele in y_test_al]
            y_pred = [int(ele[0][0]) for ele in y_pred]

        metric_scores(y_test_al, y_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type = str
    )

    parser.add_argument(
        "--model",
        type = str
    )

    args = parser.parse_args()

    predict(
        path = args.path,
        model = args.model
    )
