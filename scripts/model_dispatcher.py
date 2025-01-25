from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch.nn.functional as F

models = {
    'xg' : XGBClassifier(eta = 0.35,
                        max_depth = 12,
                        n_estimators = 1000
                        ),
    'lr': LogisticRegression(penalty = 'l2',
                            C = 1.2
                            ),
    'dt': DecisionTreeClassifier(criterion = 'gini',
                                max_depth = None
                                )
}

class dnn(nn.Module):
    def __init__(self):
        super(dnn, self).__init__()

        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

        self.dropout1 = nn.Dropout(p = 0.2)        
        self.dropout2 = nn.Dropout(p = 0.3)
        self.batchn1 = nn.BatchNorm1d(num_features = 64)
        self.batchn2 = nn.BatchNorm1d(num_features = 128)

    def forward(self, inputs):

        t = self.fc1(inputs)
        t = F.relu(t)
        t = self.batchn1(t)
        t = self.dropout1(t)
        t = self.fc2(t)
        t = F.relu(t)
        t = self.batchn2(t)
        t = self.dropout2(t)
        t = self.fc3(t)
        t = F.relu(t)
        t = self.out(t)

        return (t)
