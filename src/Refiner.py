import torch
from einops import repeat
from torch import nn

from src.submodules.ResNet import ResnetBlock


class Refiner(nn.Module):
    def __init__(self, time_steps=1, hidden_dim=64, dropout_rate: float = 0.1):
        super().__init__()
        
        assert hidden_dim % time_steps == 0

        self.kinematic_coupling = ResnetBlock(
            dim=2 * time_steps,
            dim_out=hidden_dim,
            dropout_rate=dropout_rate
        )
        
        self.kc_norm = nn.GroupNorm(
            min(32, time_steps), hidden_dim
        )
        
        self.drift_corrector = ResnetBlock(
            dim=2 * time_steps,
            dim_out=hidden_dim,
            dropout_rate=dropout_rate
        )
        self.drift_norm = nn.GroupNorm(
            min(32,  time_steps), hidden_dim
        )
        
        
        
        mixer_channels = 2 * time_steps + 2* hidden_dim
        
        groups = time_steps
        self.mixer = nn.Sequential(
            nn.GroupNorm(groups, mixer_channels),
            ResnetBlock(mixer_channels, hidden_dim, dropout_rate=dropout_rate, groups=groups),
            nn.GroupNorm(groups, hidden_dim),
            ResnetBlock(hidden_dim, hidden_dim, dropout_rate=dropout_rate,  groups=groups),
        )
        
        self.out_conv = nn.Conv2d(hidden_dim, time_steps, kernel_size=1)

        self.init_weight()
        

    def init_weight(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                
                if m is self.out_conv:
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
            self,
            A_guide: torch.Tensor,
            D_guide: torch.Tensor,
            AD_guide: torch.Tensor,
            last_frame: torch.Tensor
        ):
        """
        A_guide, D_guide, AD_guide: [B, T, H, W]
        last_frame: [B, 1, H, W]
        """
        assert A_guide.shape == D_guide.shape
        assert D_guide.shape == AD_guide.shape
        

        B, T, H, W = A_guide.shape
        
        # last frame
        last_frame = repeat(last_frame, "b 1 h w -> b t h w", t=T)

        
        # Kinematic Coupling
        ad = torch.cat([A_guide, D_guide], dim=1)
        AD_interact = self.kinematic_coupling(ad)
        AD_interact = self.kc_norm(AD_interact)
        
        
        # Drift Correction
        drift_correction = torch.cat([AD_guide, last_frame], dim=1)
        drift_correction = self.drift_corrector(drift_correction)
        drift_correction = self.drift_norm(drift_correction)
        
        # Mixer
        mixer_in = torch.cat([
            AD_guide,
            A_guide,
            AD_interact,
            drift_correction
        ], dim=1)
        
        
        delta = self.mixer(mixer_in)
        
        delta = self.out_conv(delta)

        return AD_guide + delta
