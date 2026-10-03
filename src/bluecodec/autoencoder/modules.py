import torch
import torch.nn as nn

class LayerNorm1d(nn.Module):
    """
    Channel-wise LayerNorm for 1D inputs.
    Input: [Batch, Channels, Time]
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x

class ConvNeXtBlock(nn.Module):
    """
    1D ConvNeXt Block with Dilation support.
    """
    def __init__(self, dim=512, intermediate_dim=2048, kernel_size=7, dilation=1, layer_scale_init_value=1e-6):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.dwconv = nn.Conv1d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=padding, 
            dilation=dilation, 
            groups=dim
        )
        self.norm = LayerNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, intermediate_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(intermediate_dim, dim, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.ones(1, dim, 1) * layer_scale_init_value)
        
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        x = x * self.gamma
        return residual + x

class CausalConv1d(nn.Conv1d):
    """Left-padded (causal) Conv1d. pad_mode: "zeros" (1.5M decoder) or "replicate" (official vocoder)."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, pad_mode="zeros", **kwargs):
        self._pad = (kernel_size - 1) * dilation
        if pad_mode not in ("zeros", "replicate"):
            raise ValueError(f"pad_mode must be 'zeros' or 'replicate', got {pad_mode!r}")
        self._pad_mode = pad_mode
        super().__init__(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            **kwargs
        )

    def forward(self, x):
        if self._pad_mode == "replicate":
            x = torch.nn.functional.pad(x, (self._pad, 0), mode="replicate")
        else:
            x = torch.nn.functional.pad(x, (self._pad, 0))
        return super().forward(x)

class CausalDWConv1d(nn.Module):
    """Wrapper so state-dict path is ``dwconv.net.weight`` (matches ONNX trace)."""

    def __init__(self, dim, kernel_size, dilation=1, pad_mode="zeros"):
        super().__init__()
        self.net = CausalConv1d(dim, dim, kernel_size=kernel_size, dilation=dilation, groups=dim, pad_mode=pad_mode)

    def forward(self, x):
        return self.net(x)


class CausalConvNeXtBlock(nn.Module):
    """
    1D Causal ConvNeXt Block with Dilation support.
    """
    def __init__(self, dim=512, intermediate_dim=2048, kernel_size=7, dilation=1, layer_scale_init_value=1e-6,
                 pad_mode="zeros"):
        super().__init__()

        self.dwconv = CausalDWConv1d(dim, kernel_size=kernel_size, dilation=dilation, pad_mode=pad_mode)
        self.norm = LayerNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, intermediate_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(intermediate_dim, dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.ones(1, dim, 1) * layer_scale_init_value)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x * self.gamma
        return residual + x
