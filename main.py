from src.utils import *
import torch
from torch.utils.data import DataLoader
from src import trainer

print("Start loading the data....")
train_data = get_data('train')
test_data = get_data('test')
dev_data = get_data('dev')

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
print('Finish loading the data....')

if __name__ == '__main__':
    test_loss = trainer.initiate(hyp_params, train_loader, valid_loader, test_loader)