import logging

import torch
import torch.optim as optim
from .datastore import DataStore
import torch.nn as nn
import asyncio

class Partition:
    def __init__(self, layers):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cpu_device = torch.device('cpu')
        self.activations = None
        if layers is not None:
            self.layers = layers.to(self.device)
            self.optimizer = optim.Adam(self.layers.parameters(), lr=0.001)
            self.dataStore = DataStore()
        else:
            self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def forward(self, data, batch_id=0):
        data = data.to(self.device)
        # Keep original activations with gradient tracking
        self.activations = data.clone().detach().requires_grad_(True)
        output = self.layers(self.activations)

        # Store only for backward pass (not needed for computation)
        self.dataStore.save(batch_id, self.activations)
        return output.cpu()  # Always return CPU tensor

    def backward(self, grad_output, batch_id):
        grad_output = grad_output.to(self.device)
        self.optimizer.zero_grad()

        # Retrieve original activations with gradient history
        activations = self.dataStore.get(batch_id)
        self.dataStore.delete(batch_id)

        # Backpropagate using original computation graph
        activations = activations.to(self.device)
        activations.retain_grad()  # Preserve gradient for input
        output = self.layers(activations)  # Reattach to computation graph

        # Compute gradients using original forward pass
        output.backward(grad_output)
        self.optimizer.step()

        # Return input gradients (from original computation)
        return activations.grad.detach().cpu()

    # if is last partition can be used
    def compute_loss_and_grad(self, predictions, labels):
        # Ensure predictions require gradient
        predictions = predictions.to(self.device)
        labels = labels.to(self.device)
        predictions = predictions.detach().requires_grad_(True)
        loss = self.loss_fn(predictions, labels)
        loss.backward()
        grad_output = predictions.grad
        return loss.item(), grad_output.cpu()

    async def getStateDict(self):
        return await asyncio.to_thread(lambda: self.layers.state_dict())

    async def load_partition_state_dict(self, state_dict):
        current_state_dict = self.layers.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in current_state_dict}
        self.layers.load_state_dict(filtered_state_dict, strict=False)
        for param in self.layers.parameters():
            param.grad = None
        print("State dictionary loaded successfully for matching keys.")


class LossCalculationPartition:
    def __init__(self):
        self.loss_fn = nn.CrossEntropyLoss()

def compute_loss_and_grad(predictions, labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss()
    predictions = predictions.to(device)
    labels = labels.to(device)
    predictions = predictions.detach().requires_grad_(True)
    loss = loss_fn(predictions, labels)
    # Compute gradient of loss w.r.t. predictions
    loss.backward()
    grad_output = predictions.grad
    return loss.item(), grad_output


LR = 2e-5
class BertPartition(nn.Module):
    def __init__(self, layers, is_first=False, is_last=False):
        super().__init__()
        self.layers = layers
        self.is_first = is_first
        self.is_last = is_last
        self.dataStore = {}  # batch_id → (inputs, extended_mask)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers = self.layers.to(self.device)

        self.optimizer = torch.optim.Adam(self.layers.parameters(), lr=LR)
        logging.critical(f"Initialized Bert Partition on {self.device}")


    def forward(self,data,batch_id=0,attention_mask=None):

        data = data.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        # Move inputs to this partition's device
        if self.is_first:
            inp = data.to(self.device)
            self.dataStore[batch_id] = (inp, None)
            out = self.layers(inp)
        else:
            hidden = data.to(self.device)
            if attention_mask is not None:
                m = attention_mask.to(self.device).bool()
                m = m.unsqueeze(1).unsqueeze(2)
                self.dataStore[batch_id] = (hidden, m)
                attn = m
            else:
                self.dataStore[batch_id] = (hidden, None)
                attn = None

            if isinstance(self.layers, nn.Sequential):
                out = self.layers(hidden)
            else:
                out = self.layers(hidden, attention_mask=attn)

        # Unpack HF tuple outputs
        if isinstance(out, tuple):
            out = out[0]

        if self.is_last:
            return out.detach()

        return out.detach()

    def backward(self, grad_output, batch_id):
        saved_inp, saved_mask = self.dataStore.pop(batch_id)
        self.optimizer.zero_grad(set_to_none=True)

        if self.is_first:
            # Embedding layer handling
            inp = saved_inp.to(self.device)
            outputs = self.layers(inp)
            core = outputs[0] if isinstance(outputs, tuple) else outputs
            self.optimizer.zero_grad()
            core.backward(gradient=grad_output.to(self.device))
            self.optimizer.step()
            del  core , saved_inp
            return torch.zeros_like(inp, dtype=torch.float32)
        else:
            # Other layers
            hidden = saved_inp.to(self.device).requires_grad_(True)
            attn_mask = saved_mask if saved_mask is not None else None

            self.optimizer.zero_grad(set_to_none=True) # for optimize memory usage
            if isinstance(self.layers, nn.Sequential):
                outputs = self.layers(hidden)
            else:
                outputs = self.layers(hidden, attention_mask=attn_mask)

            core = outputs[0] if isinstance(outputs, tuple) else outputs
            core.backward(gradient=grad_output.to(self.device))
            self.optimizer.step()

            grad_result = hidden.grad.detach().clone()
            del hidden , outputs ,core , saved_inp , attn_mask
            return grad_result

    async def getStateDict(self):
        return await asyncio.to_thread(lambda: self.layers.state_dict())

    async def load_partition_state_dict(self, state_dict):

        current_state_dict = self.layers.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in current_state_dict}
        self.layers.load_state_dict(filtered_state_dict, strict=False)
