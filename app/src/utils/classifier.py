import torch
import torch.nn as nn
from typing import List

class TextClassifier(nn.Module):
    def __init__(self, input_size: int = 1024, hidden_sizes: List[int] = [512, 256],
                 num_classes: int = 10, dropout: float = 0.5):
        super(TextClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, num_classes))
        self.layer_stack = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)