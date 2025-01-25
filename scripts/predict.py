import torch
import joblib
import argparse
import pandas as pd
import model_dispatcher
from config_loader import config
from torch.utils.data import DataLoader
import src.dataset as dataset
from src.eval_metrics import metric_scores
from preprocessing import data_preprocessing

def predict(path, model):

    print ("INGESTING THE DATA")
    df = pd.read_csv(path)

    print ("RUNNING DATA PREPROCESSING")
    df = data_preprocessing(df)

    if model in ['xg', 'lr', 'dt']:

        X_test = df.drop(columns = ['label'])
        y_test = df.label.values

        print ("LOADING THE MODEL")
        mod = joblib.load(config['OUTPUTS']['MODEL_OUTPUT'] + "/" + str(model) + '.bin')
        
        print ("PREDICTING...")
        predictions = mod.predict(X_test)

        print ("EVALUATING THE PERFORMANCE")
        metric_scores(y_test, predictions)
    
    elif model == 'dnn':
        model = model_dispatcher.dnn()
        model.to(config['DEVICE'])

        print ("LOADING THE MODEL")
        model.load_state_dict(torch.load(config['OUTPUTS']['MODEL_OUTPUT'] + '/dnn.pth'))

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

        print ("PREDICTING...")
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

        print ("EVALUATING THE PERFORMANCE")
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
