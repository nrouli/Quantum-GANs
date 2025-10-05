import torch.nn as nn
from torch import Tensor

class Discriminator(nn.Module):
    def __init__(self, hidden_size=8, dropout= 0.4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(64, hidden_size),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        
        self._initialize_weights()
       
    def _initialize_weights(self):
        """Gentler initialization to give generator a chance"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Smaller std
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        return self.model(x.view(batch_size, -1))
    
    
    