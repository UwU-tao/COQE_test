import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

f1 = F1Score(task='binary', average='macro', threshold=0.5, num_classes=2)
recall = Recall(task='binary', average='macro', threshold=0.5, num_classes=2)
precision = Precision(task='binary', average='macro', threshold=0.5, num_classes=2)
accuracy = Accuracy(task='binary', average='macro', threshold=0.5, num_classes=2)
f1.to(device)
recall.to(device)
precision.to(device)
accuracy.to(device)

def metrics(results, truths):
    f1_score = f1(results, truths)
    recall_score = recall(results, truths)
    precision_score = precision(results, truths)
    acc_score = accuracy(results, truths)

    return acc_score, precision_score, recall_score, f1_score