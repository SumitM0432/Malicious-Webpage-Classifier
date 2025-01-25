import pandas as pd
import numpy as np
from sklearn import metrics
import yaml

with open("Config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

df = pd.read_csv(config['paths']['TRAINING_FILE'])
print (df.shape)

df.replace({'good' : 0, 'bad' : 1}, inplace = True)

arr = np.zeros((df.shape[0],), dtype = np.int64).tolist()

acc = metrics.accuracy_score(np.array(df['label']).tolist(), arr)
print (acc)

