import torch
import torch.nn as nn

class LateralConnection(nn.Module):
    """
    Implements lateral connections between neurons in the same layer.
    Allows for competitive dynamics (lateral inhibition) or 
    cooperative dynamics (lateral excitation).
    """
    def __init__(self, n_units, connection_type='inhibitory', strength=0.1):
        """
        Args:
            n_units: Number of neurons in the layer
            connection_type: 'inhibitory' or 'excitatory'
            strength: Connection strength
        """
        super().__init__()
        self.n_units = n_units
        self.connection_type = connection_type
        self.strength = strength
        
        # Create lateral connection matrix 
        self.lateral_weights = nn.Parameter(
            torch.randn(n_units, n_units) * 0.01
        )
        
        # Create mask to exclude self-connections (diagonal = 0)
        self.register_buffer('mask', torch.ones(n_units, n_units) - torch.eye(n_units))
        
    def forward(self, x):
        """
        Apply lateral connections to layer activations.
        
        Args:
            x: Layer activations of shape (batch_size, seq_len, n_units)
            
        Returns:
            x_lateral: Activations after lateral influence
        """
        # Apply mask to remove self-connections
        masked_weights = self.lateral_weights * self.mask
        
        if self.connection_type == 'inhibitory':
            masked_weights = -torch.abs(masked_weights) * self.strength
        else:  # excitatory
            masked_weights = torch.abs(masked_weights) * self.strength
            
        # Compute lateral influence
        lateral_influence = torch.matmul(x, masked_weights)
        
        # Add lateral influence to original activations
        x_lateral = x + lateral_influence
        
        return x_lateral