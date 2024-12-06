import torch
import torch.nn as nn
import torch.nn.functional as F
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
        super(LoRAedLinear, self).__init__()
        in_features = dense.in_features
        out_features = dense.out_features
        self.dense = dense
        if drop_out > 0.:
            self.lora_dropout = nn.Dropout(p=drop_out)
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        if rank > 0:
            self.lora_A = nn.Parameter(self.dense.weight.new_zeros((rank, in_features),dtype=torch.float32))
            self.lora_B = nn.Parameter(self.dense.weight.new_zeros((out_features, rank),dtype=torch.float32))
            self.scaling = alpha / rank
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor):
        result = self.dense(x)           
        return result + (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling

    
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

        if not isinstance(kernel_size, int):
            kernel_size = kernel_size[0]
        # Actual trainable parameters
        if rank > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((in_channels * kernel_size, rank * kernel_size ),dtype=torch.float32)
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((rank * kernel_size, out_channels//self.conv.groups*kernel_size),dtype=torch.float32)
            )
            self.scaling = alpha / rank
            # Freezing the pre-trained weight matrix
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
            self.conv.weight + (self.lora_A @ self.lora_B).view(self.conv.weight.shape) * self.scaling,
            self.conv.bias
        )
    
class LoRAedconvT(nn.Module):
    def __init__(self,
                conv: nn.ConvTranspose2d,
                rank: int=4,
                alpha: float=1.0):
        super(LoRAedconvT, self).__init__()
        
        kernel_size = conv.kernel_size
        in_channels = conv.in_channels
        out_channels = conv.out_channels
        self.conv = conv

        if not isinstance(kernel_size, int):
            kernel_size = kernel_size[0]
        # Actual trainable parameters
        if rank > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((in_channels * kernel_size, rank * kernel_size ),dtype=torch.float32)
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((rank * kernel_size, out_channels//self.conv.groups*kernel_size),dtype=torch.float32)
            )
            self.scaling = alpha / rank
            # Freezing the pre-trained weight matrix

        import pdb
        # pdb.set_trace()
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        num_spatial_dims = 2
        output_padding = self.conv._output_padding(
            x,
            None,
            self.conv.stride,  # type: ignore[arg-type]
            self.conv.padding,  # type: ignore[arg-type]
            self.conv.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.conv.dilation,  # type: ignore[arg-type]
        )

        return F.conv_transpose2d(
            x,
            self.conv.weight + (self.lora_A @ self.lora_B ).view(self.conv.weight.shape) * self.scaling,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            output_padding,
            self.conv.groups,
            self.conv.dilation,
        )



def LoRA_injection(model:nn.Module, type, rank, scale=1.0):
    for name, block in model.named_children():
        if isinstance(block, nn.Linear) and 'linear' in type:
            block = LoRAedLinear(block, rank, scale)
            setattr(model, name, block)
        
        elif isinstance(block, nn.Conv2d) and 'conv' in type:
            min_channel = min(block.in_channels, block.out_channels)
            if min_channel > rank:
                block = LoRAedconv(block, rank, scale)
                setattr(model, name, block)

        elif isinstance(block, nn.ConvTranspose2d) and 'convT' in type:
            min_channel = min(block.in_channels, block.out_channels)
            if min_channel > 4:
                block = LoRAedconvT(block, rank, scale)
                setattr(model, name, block)

        else:
            LoRA_injection(block, type, rank)