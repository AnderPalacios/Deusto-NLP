import numpy as np
import torch
import torch.nn as nn

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LSTMDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels.astype(np.int64) 
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx]), torch.tensor(self.labels[idx])



class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)  
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # For embeddings, seq_len=1, so we add a dummy dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :] # Take the last timestep
        out = self.dropout(out)
        out = self.fc(out)
        return out
