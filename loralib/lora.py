import torch
import torch.nn as nn
from layers import ConvLoRA
import math


class LoRAedLinear(nn.Module):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        dense: nn.Linear,
        rank:int = 4,
        drop_out:float = 0,
        alpha:float=1.0,
    ):
        in_features = dense.in_features
        out_features = dense.out_features
        self.dense = dense
        if drop_out > 0.:
            self.lora_dropout = nn.Dropout(p=drop_out)
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        if rank > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((rank, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, rank)))
            self.scaling = alpha / rank
            # Freezing the pre-trained weight matrix
        self.dense.requires_grad_=False
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor):
        result = self.dense(x)           
        result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result
    
class LoRAedconv(nn.Module):
    def __init__(self,
                conv: nn.Conv2d,
                rank: int=4,
                alpha: float=1.0):
        super(LoRAedconv, self).__init__()
        
        kernel_size = conv.kernel_size
        in_channels = conv.in_channels
        out_channels = conv.out_channels
        self.conv = conv

        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if rank > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((rank * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, rank * kernel_size))
            )
            self.scaling = alpha / self.r
            # Freezing the pre-trained weight matrix
        self.conv.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.conv._conv_forward(
            x, 
            self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
            self.conv.bias
        )


def LoRA_injection(model:nn.Module, type):
    for name, block in model.named_children():
        if isinstance(block, nn.Linear) and type == 'linear':
            block = LoRAedLinear(block, 4, 1)
            setattr(model, name, block)
        
        elif isinstance(block, nn.Conv2d) and type == 'conv':
            min_channel = min(block.in_channels, block.out_channels)
            if min_channel > 4:
                block = ConvLoRA(block, 4, 1)
                setattr(model, name, block)

        else:
            LoRA_injection(block, type)