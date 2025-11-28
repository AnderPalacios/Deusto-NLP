import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CNNDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels.astype(np.int64)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx]), torch.tensor(self.labels[idx])

class CNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_filters=128, kernel_sizes=[3,4,5], dropout=0.5):
        super().__init__()
        
        # embedding_dim becomes the "sequence length", features=1
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1,          # only 1 channel
                      out_channels=num_filters,
                      kernel_size=k) 
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch_size, embedding_dim)
        x = x.unsqueeze(1)                    # → (B, 1, embedding_dim)  <-- channels=1
        convs = []
        for conv in self.convs:
            out = conv(x)                     # → (B, num_filters, L_out)
            out = torch.relu(out)
            out = torch.max_pool1d(out, out.size(2)).squeeze(2)  # global max → (B, num_filters)
            convs.append(out)
        
        x = torch.cat(convs, dim=1)           # → (B, num_filters * len(kernel_sizes))
        x = self.dropout(x)
        x = self.fc(x)
        return x
