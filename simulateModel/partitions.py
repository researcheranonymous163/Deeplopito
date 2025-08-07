import torch.nn as nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class FinalPartition:
    def __init__(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss_and_grad(self, predictions, labels):
        # Ensure predictions require gradient

        predictions = predictions.to(device)
        labels = labels.to(device)
        predictions = predictions.detach().requires_grad_(True)

        loss = self.loss_fn(predictions, labels)
        # Compute gradient of loss w.r.t. predictions
        loss.backward()
        grad_output = predictions.grad
        return loss.item(), grad_output.cpu()
   
