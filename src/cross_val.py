import pandas as pd
from sklearn import model_selection

def create_folds(df):

    df["kfold"] = -1

    # shuffling
    df = df.sample(frac = 1).reset_index(drop = True)

    # targets
    target = df.label.values

    # stratified k fold with 5 folds
    kf = model_selection.StratifiedKFold(n_splits = 5)

    for i, (_, val) in enumerate(kf.split(X = df, y = target)):
        df.loc[val, 'kfold'] = i

    return df

