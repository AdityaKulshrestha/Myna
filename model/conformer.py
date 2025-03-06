import torch 
# from habana_frameworks.t as htcore
from torchaudio.models import Conformer

class ConformerAcousticModel(torch.nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            num_heads: int, 
            ffn_dim: int, 
            num_layers: int, 
            depthwise_conv_kernel_size: int, 
            dropout: float=0.0, 
            convolution_first: bool = False
    ):
        # super().__init__()
        self.model = Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            convolution_first=convolution_first
        )

    
        