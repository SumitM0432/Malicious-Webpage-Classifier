import pandas as pd
import numpy as np
import config
from sklearn import metrics

df = pd.read_csv(config.TRAINING_FILE)
print (df.shape)

df.replace({'good' : 0, 'bad' : 1}, inplace = True)

arr = np.zeros((df.shape[0],), dtype = np.int64).tolist()

acc = metrics.accuracy_score(np.array(df['label']).tolist(), arr)
print (acc)

# 0.9772891666666667