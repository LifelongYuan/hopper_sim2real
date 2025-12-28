import torch
import torch.nn as nn
import numpy as np

class RNDNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, rnd_reward_scale=0.1):
        """Random Network Distillation module.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
        """
        super().__init__()
        
        # Fixed random network (target)
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Initialize target network with orthogonal initialization
        for module in self.target.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # Freeze target network weights
        for param in self.target.parameters():
            param.requires_grad = False
        # Predictor network that tries to match the random network
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize target network with random weights and freeze them
        for param in self.target.parameters():
            param.requires_grad = False
        self.rnd_reward_scale = rnd_reward_scale
    def forward(self, x):
        """
        Forward pass through both networks.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            prediction_error: MSE between target and predictor outputs
            predictor_features: Features from predictor network
        """
        with torch.no_grad():
            target_features = self.target(x)
        
        predictor_features = self.predictor(x)
        
        # Compute prediction error (intrinsic reward)
        prediction_error = torch.mean((target_features - predictor_features) ** 2, dim=-1)
        
        return prediction_error, predictor_features

    def compute_intrinsic_reward(self, x):
        """Compute intrinsic reward from prediction error.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            intrinsic_reward: Prediction error as intrinsic reward
        """
        prediction_error, _ = self.forward(x)
        return prediction_error