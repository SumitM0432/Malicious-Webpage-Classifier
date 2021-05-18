import torch
import dataset
import joblib
import config
from preprocessing import preprocessing
import pandas as pd
import argparse
import Metrics
import model_dispatcher
from torch.utils.data import DataLoader

def predict(path, model):

    df = pd.read_csv(path)

    df = preprocessing(df)

    if model in ['xg', 'lr', 'dt']:

        X_test = df.drop(columns = ['label'])
        y_test = df.label.values

        mod = joblib.load(config.MODEL_OUTPUT + str(model) + '.bin')
        
        predictions = mod.predict(X_test)

        Metrics.metric_scores(y_test, predictions)
    
    elif model == 'dnn':
        model = model_dispatcher.dnn()
        model.to(config.DEVICE)

        model.load_state_dict(torch.load(config.MODEL_OUTPUT + '/dnn.pth'))

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
                X_test = X_test.to(config.DEVICE)

                predictions = model(X_test.float())
                pred = torch.round(torch.sigmoid(predictions))

                y_test_al.append(y_test.tolist())
                y_pred.append(pred.tolist())

            # Changing the Predictions into list 
            y_test_al = [ele[0] for ele in y_test_al]
            y_pred = [int(ele[0][0]) for ele in y_pred]

        Metrics.metric_scores(y_test_al, y_pred)

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
