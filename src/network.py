import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, in_dim: int, nf: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        # Network architecture adapted for HIV state space
        self.layers = nn.Sequential(
            nn.Linear(in_dim, nf), 
            nn.LeakyReLU(),
            nn.Linear(nf, nf), 
            nn.LeakyReLU(),
            nn.Linear(nf, nf), 
            nn.LeakyReLU(),
            nn.Linear(nf, out_dim)
        )

        self.init_weights()

    def init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)
