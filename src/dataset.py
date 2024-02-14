import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd

class MyDataset(Dataset):

    def __init__(self, split):
        texts = []
        labels = []
        
        with open(f"./data/smartphone/{split}.txt") as f:
            lines = f.readlines()
            for line in lines:
                temp = line.split("\t")
                if len(temp) == 2:
                    text, label = temp
                    texts.append(text)
                    labels.append(int(label))
            
        self.data_dict = pd.DataFrame({'text': texts, 'label': labels})                 
        self.num_classes = 2

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.data_dict.iloc[idx,0]
        label = self.data_dict.iloc[idx,1]
        
        sample = {'text': text, 'label': torch.tensor(label, dtype=torch.long)}

        return sample