import torch
import torch.nn as nn
from src import models
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from src.utils import *

def initiate(train_loader, valid_loader, test_loader):
    device = torch.device('cuda')
    
    bert = AutoModel.from_pretrained("vinai/phobert-base-v2")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    bert.to(device)
    
    model = getattr(models, 'Simple')(input_dim=768, hidden_dim=256, output_dim=2)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # Fixed to use the model's parameters
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    settings = {
        'model': model,
        'device': device,
        'bert': bert,
        'tokenizer': tokenizer,
        'optimizer': optimizer,
        'criterion': criterion,
        'scheduler': scheduler,
    }
    
    return train_model(settings, train_loader, valid_loader, test_loader)

def train_model(settings, train_loader, valid_loader, test_loader):
    model = settings['model']
    bert = settings['bert']
    tokenizer = settings['tokenizer']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    
    def train(model, bert, tokenizer, optimizer, criterion):
        model.train()
        epoch_loss = 0
        results = []
        truth = []
        
        for batch in tqdm(train_loader):
            text = batch['text']
            label = batch['label']
            label = label.to(settings['device'])  # Move label to the same device as the model
            
            optimizer.zero_grad()
            text_encoded = tokenizer(text, padding=True, return_tensors='pt').to(settings['device'])
            
            with torch.no_grad():
                outs = bert(**text_encoded)
            
            predictions = model(outs.pooler_output)
            preds = predictions
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(train_loader)
            results.append(preds)
            truth.append(label)
        
        results = torch.cat(results)
        truth = torch.cat(truth)
        return results, truth, epoch_loss / len(train_loader)
    
    def evaluate(model, bert, tokenizer, criterion):
        model.eval()
        epoch_loss = 0
        results = []
        truth = []
        
        with torch.no_grad():
            for batch in tqdm(valid_loader):
                text = batch['text']
                label = batch['label']
                label = label.to(settings['device'])  # Move label to the same device as the model
                
                text_encoded = tokenizer(text, padding=True, return_tensors='pt').to(settings['device'])
                with torch.no_grad():
                    outs = bert(**text_encoded)
                
                predictions = model(outs.pooler_output)
                preds = predictions
                loss = criterion(predictions, label)
                epoch_loss += loss.item() * len(valid_loader)

                results.append(preds)
                truth.append(label)
                
        results = torch.cat(results)
        truth = torch.cat(truth)
        return results, truth, epoch_loss / len(valid_loader)
    
    best_valid_loss = float('inf')
    for epoch in range(1, 20):
        train_results, train_truth, train_loss = train(model, bert, tokenizer, optimizer, criterion)
        valid_results, valid_truth, valid_loss = evaluate(model, bert, tokenizer, criterion)
        scheduler.step()

        # Metrics calculation function is not provided. Please implement or replace it accordingly.
        train_metrics = calculate_metrics(train_results, train_truth)
        valid_metrics = calculate_metrics(valid_results, valid_truth)

        if epoch == 1:
            print(f'Epoch  |     Train Loss     |     Train Accuracy     |     Valid Loss     |     Valid Accuracy     |     Precision     |     Recall     |     F1-Score     |')
            
        print(f'{epoch:^7d}|{train_loss:^20.4f}|{train_acc:^24.4f}|{valid_loss:^20.4f}|{valid_acc:^24.4f}|{valid_metrics["precision"]:^19.4f}|{valid_metrics["recall"]:^16.4f}|{valid_metrics["f1"]:^18.4f}|')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_model(model, 'best_model.pth')
        
        if test_loader is not None:
            # Assuming you have a function load_model to load the model and calculate_metrics for test data
            loaded_model = load_model(model, 'best_model.pth')
            test_results, test_truth, test_loss, test_acc = evaluate(loaded_model, bert, tokenizer, criterion)
            test_metrics = calculate_metrics(test_results, test_truth)
        
            print("\n\nTest Acc {:5.4f} | Test Precision {:5.4f} | Test Recall {:5.4f} | Test F1-score {:5.4f}".format(test_acc, test_metrics["precision"], test_metrics["recall"], test_metrics["f1"]))
