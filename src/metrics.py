import torch
import numpy as np
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryAccuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

f1 = BinaryF1Score()
recall = BinaryRecall()
precision = BinaryPrecision()
accuracy = BinaryAccuracy()
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