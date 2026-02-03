import torch
from einops import rearrange, repeat

from torch.nn import Module, Conv2d
import torch.nn as nn
from utils.wavelet_transform import WaveletTransform
from src.submodules.ResNet import DilatedResnetBlock



class TemporalMixer(nn.Module):
    
    
    def __init__(self, dim:int):
        super().__init__()
        
        self.entry_proj = nn.Linear(dim, dim)
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, groups=dim),
                nn.GELU(),
                nn.Conv1d(dim, dim, kernel_size=1)
            ) for dilation in (1, 2, 4)
        ])
        
        self.fusion_conv = nn.Conv1d(dim, dim, kernel_size=1)
        
    
    
    def forward(self, x:torch.Tensor):
        # x: (B, dim, h, w)
        
        b, d, h, w = x.shape
        
        x = rearrange(x, "b d h w -> b h w d")
        x = self.entry_proj(x)
        
        x = rearrange(x, "b h w d -> b d (h w)")
        
        residual = x
        
        for layer in self.layers:
            x = layer(x) + residual
            residual = x
        
        
        x = self.fusion_conv(x) # b h w d
        
        return rearrange(x, "b d (h w)  -> b d h w", h=h, w=w)
        
        

        

class SpatioTemporalBlock(nn.Module):
    
    
    def __init__(self, dim: int, dilation: int, drop_rate: float = 0.05):
        super().__init__()
        
        
        self.temporal_mlp = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1)
        )
        
        self.dropout = nn.Dropout(drop_rate)
        
        self.spatial_mixer = DilatedResnetBlock(dim, dim, dilation=dilation)
        
 

    
    def forward(self, x: torch.Tensor):

        
        b,d,h,w= x.shape
        
        spatial = self.spatial_mixer(x)
        
        temporal = rearrange(spatial, "b d h w -> b d (h w)")
        temporal = self.temporal_mlp(temporal)
        
        temporal = self.dropout(temporal)
        
        temporal = rearrange(temporal, "b d (h w) -> b d h w", h=h, w=w)
        
        return x + temporal
        
        
        




class ApproximationNetwork(Module):

    def __init__(self, hidden_size: int, dropout_rate: float = 0.1, timesteps : int = 6, cell_numbers: int = 3) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.time_steps = timesteps
        self.cell_numbers = cell_numbers
        


    def init_weight(self):
        
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",  nonlinearity="relu")
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            # elif isinstance(m , (nn.GroupNorm, nn.LayerNorm)):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
        
        nn.init.constant_(self.decoder.weight, 0)
        nn.init.constant_(self.decoder.bias, 0)

    def dummy_run(self, data: torch.Tensor, wavelet: WaveletTransform):
        n_layers = self.cell_numbers
        
        # group norm
        
        if self.hidden_size % 32 == 0:
            groups = 32
        elif self.hidden_size % 16 == 0:
            groups = 16
        elif self.hidden_size % 8 == 0:
            groups = 8
        else:
            groups = 1
        
        
        data = wavelet.transform(data)

        approxi_coeff = data["A"]

        B, T, h, w = approxi_coeff.shape
        
        assert 2 ** (n_layers - 1) < h

        self.encoder = nn.Conv2d(self.time_steps, self.hidden_size, kernel_size=3, padding=1)
        
        self.temporal_injector = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=self.hidden_size // 4, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GELU(),
            nn.Conv3d(in_channels=self.hidden_size //4, out_channels=self.hidden_size // 4, kernel_size=(self.time_steps, 1, 1))
            
        )
        
        self.fusion = nn.Conv2d(self.hidden_size // 4 + self.hidden_size, self.hidden_size, kernel_size=1)
        

        self.norm_before = nn.GroupNorm(groups, self.hidden_size)
        self.mixer_layers = nn.Sequential(*[
            SpatioTemporalBlock(
                dim=self.hidden_size,
                dilation=2**i
            )
            for i in range(n_layers)
        ])
        
        self.norm_after = nn.GroupNorm(groups, self.hidden_size)
        self.decoder = nn.Conv2d(self.hidden_size, self.time_steps, kernel_size=1)

        self.init_weight()

    def forward(self, data: torch.Tensor, wavelet: WaveletTransform):

        coeffs = wavelet.transform(data)

        # B, T, H, W
        origin_a = coeffs["A"]
        B, T, H, W = origin_a.shape
        
        x = self.encoder(origin_a)
        
        x = self.norm_before(x)
        
        # temporal
        temporal = origin_a.unsqueeze(1) # b 1 t h w
        temporal = self.temporal_injector(temporal).squeeze(2) # b, d//4, 1, h, w
        
        

        x = torch.cat([x, temporal], dim=1)
        
        x = self.fusion(x)
        
        x = self.mixer_layers(x)
        
        
        x = self.norm_after(x)
        x = self.decoder(x)
        
        coeffs["A"] = x + origin_a

        # repeat the last frame
        for i in range(wavelet.level):
            detail = coeffs[f"D{i + 1}"]  # (B, T, 3, h, w)

            # only use the last frame
            detail = detail[:, -1:, :, :, :]  # (B, 1, 3, h, w)
            # repeat T times
            detail = detail.repeat(1, T, 1, 1, 1)  # (B, T, 3, h, w)
            coeffs[f"D{i + 1}"] = detail

        reconstructed = wavelet.reverse(coeffs)

        return reconstructed.contiguous(), coeffs
