import torch
from torch.utils.data import Dataset

class MaliciousBenignData(Dataset):
    def __init__(self, df):
        self.df = df
        self.input = self.df.drop(columns = ['label']).values
        self.target = self.df.label
        
    def __len__(self):
        return (len(self.df))
    
    def __getitem__(self, idx):
        return (torch.tensor(self.input[idx]), torch.tensor(self.target[idx]))


