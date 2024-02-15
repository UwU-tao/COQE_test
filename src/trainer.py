import torch
import torch.nn as nn
from src import models
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from src.utils import *
from src.metrics import metrics

def initiate(train_loader, valid_loader, test_loader):
    device = torch.device('cuda')
    
    bert = AutoModel.from_pretrained("vinai/phobert-base-v2")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    bert.to(device)
    
    model = getattr(models, 'Simple')(input_dim=768, hidden_dim=256, output_dim=1)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # Fixed to use the model's parameters
    criterion = nn.BCEWithLogitsLoss()
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
            label = label.to(settings['device'])
            
            optimizer.zero_grad()
            text_encoded = tokenizer(text, padding=True, return_tensors='pt').to(settings['device'])
            input_ids = text_encoded['input_ids']
            attention_mask = text_encoded['attention_mask']
            # with torch.no_grad():
            outs = bert(input_ids=input_ids, attention_mask=attention_mask)
            
            predictions = model(outs.pooler_output).squeeze(1)
            preds = predictions
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
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
            for batch in valid_loader:
                text = batch['text']
                label = batch['label']
                label = label.to(settings['device'])
                
                text_encoded = tokenizer(text, padding=True, return_tensors='pt').to(settings['device'])
                input_ids = text_encoded['input_ids']
                attention_mask = text_encoded['attention_mask']
                # with torch.no_grad():
                outs = bert(input_ids=input_ids, attention_mask=attention_mask)
                
                predictions = model(outs.pooler_output).squeeze(1)
                preds = predictions
                loss = criterion(predictions, label)
                epoch_loss += loss.item()

                results.append(preds)
                truth.append(label)
                
        results = torch.cat(results)
        truth = torch.cat(truth)
        return results, truth, epoch_loss / len(valid_loader)
    
    best_valid_loss = float('inf')
    for epoch in range(1, 10):
        train_results, train_truth, train_loss = train(model, bert, tokenizer, optimizer, criterion)
        valid_results, valid_truth, valid_loss = evaluate(model, bert, tokenizer, criterion)
        scheduler.step()

        train_acc, train_prec, train_recall, train_f1 = metrics(train_results, train_truth)
        val_acc, val_prec, val_recall, val_f1 = metrics(valid_results, valid_truth)

        if epoch == 1:
            print(f'Epoch  |     Train Loss     |     Train Accuracy     |     Valid Loss     |     Valid Accuracy     |     Precision     |     Recall     |     F1-Score     |')
            
        print(f'{epoch:^7d}|{train_loss:^20.4f}|{train_acc:^24.4f}|{valid_loss:^20.4f}|{val_acc:^24.4f}|{val_prec:^19.4f}|{val_recall:^16.4f}|{val_f1:^18.4f}|')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_model(model, 'best_model')
        
    if test_loader is not None:
        loaded_model = load_model('best_model')
        test_results, test_truth, test_loss = evaluate(loaded_model, bert, tokenizer, criterion)
        test_acc, test_prec, test_recall, test_f1 = metrics(test_results, test_truth)
    
        print("\n\nTest Acc {:5.4f} | Test Precision {:5.4f} | Test Recall {:5.4f} | Test f1-score {:5.4f}".format(test_acc, test_prec, test_recall, test_f1))