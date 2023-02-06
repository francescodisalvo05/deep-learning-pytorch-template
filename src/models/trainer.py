from datetime import datetime

import numpy as np
import torch
import os


class Trainer:

    def __init__(self, model, params):
        self.model = model
        self.criterion = params['loss']
        self.output_path = params['store_ckpt_path']
        self.device = params['device']
        self.lr = params['lr']

        self.optimizers = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    
    def train(self, model, train_loader, epoch):

        train_loss, correct, train_metrics = 0.0, 0, {}
        model = model.float().to(self.device)

        model.train()

        for idx, (inputs,labels) in enumerate(train_loader):

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = model(inputs.float())

            loss = self.criterion(outputs,labels)
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct += torch.sum(labels == predicted).item()

            loss.backward()
            self.optimizer.step()

        train_metrics['loss'] = train_loss / (idx + 1)
        train_metrics['accuracy'] = 100. * correct / len(train_loader.dataset)
        
        return train_metrics

    
    def validate(self, model, val_loader):

        val_loss, correct, metrics = 0.0, 0, {}
        model = model.float().to(self.device)
        model.eval()

        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(val_loader):

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs.float())
                val_loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += torch.sum(labels == predicted).item()

        metrics['loss'] = val_loss / (idx + 1)
        metrics['accuracy'] = 100. * correct / len(val_loader.dataset)

        return metrics

    
    def store_checkpoint(self, epoch, store_ckpt_path, train_metrics, val_metrics):

        filename = f"{epoch}-{train_metrics['loss']} - {val_metrics['loss']}-last.pt"
        path = os.path.join(store_ckpt_path,filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': train_metrics['loss'],
        }, path)

    
    def load_checkpoint(self, load_ckpt_filepath):
        checkpoint = torch.load(load_ckpt_filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        
        # loss = checkpoint['loss'] ? 
        return epoch