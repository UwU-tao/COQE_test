from src.dataset import MyDataset
import torch

def get_data(split='train'):
    data = MyDataset(split)
    return data

def save_model(model, name=''):
    name = name if len(name) > 0 else 'default_model'
    torch.save(model, f'pretrained/{name}.pt')


def load_model(name=''):
    name = name if len(name) > 0 else 'default_model'
    model = torch.load(f'pretrained/{name}.pt')
    return model