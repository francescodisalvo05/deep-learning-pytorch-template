import torch


class Accuracy(torch.nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()
    
    def forward(self, predictions, targets):
        """Example with accuracy"""

        _, predictions = torch.max(predictions, dim=1)
        correct = (predictions == targets).sum().float()
        total = targets.numel()
        accuracy = correct / total
        return accuracy