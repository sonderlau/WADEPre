import torch
from einops import rearrange, repeat

from src.submodules.FPN import FPN
from torch.nn import Module
import torch.nn as nn

from utils.wavelet_transform import WaveletTransform, WaveletCoeffDict



class DetailNetwork(Module):
    def __init__(
        self,
        fpn_time: int = 6,
        idr_dim: int = 32,
        feature_channel: int = 128,
        layer_channels=[64, 128, 256],
        num_blocks=[2, 2, 2],
        dropout_rate : float = 0.1,
    ):
        super(DetailNetwork, self).__init__()
        self.fpn = None

        assert len(num_blocks) == len(
            layer_channels
        ), "num_blocks and layer_channels must have the same length."

        assert idr_dim % 4 == 0, "idr_dim must be divisible by 4."

        self.fpn_time = fpn_time
        self.feature_channel = feature_channel
        self.layer_channels = layer_channels
        self.num_blocks = num_blocks
        self.idr_dim = idr_dim
        self.dropout_rate = dropout_rate

    def init_weight(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "idr" in name and any(x in name for x in [".3", "weight"]):
                    nn.init.normal_(m.weight, mean=0, std=0.001)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def dummy_run(self, data: torch.Tensor, wavelet: WaveletTransform):
        """Dummy run to determine the shape

        Args:
            data (torch.Tensor): the input data
            wavelet (WaveletTransform): the wavelet transform module
        """

        coeffs = wavelet.transform(data)
        level = wavelet.level

        input_sizes = []

        for i in range(level):
            detail = coeffs[f"D{i + 1}"]
            h, w = detail.shape[-2:]
            input_sizes.append((h, w))

        self.fpn = FPN(
            num_blocks=self.num_blocks,
            in_channels=self.fpn_time,
            feature_channels=self.fpn_time,
            layer_channels=self.layer_channels,
            input_sizes=input_sizes,
        )

        self.temporal_mlp_before = nn.Sequential(
            nn.Linear(self.fpn_time, self.feature_channel),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_channel, self.fpn_time),  # 输出 T' 步预测
        )

        self.temporal_mlp_after = nn.Sequential(
            nn.Linear(self.fpn_time, self.feature_channel),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_channel, self.fpn_time),  # 输出 T' 步预测
        )

        # Iterative Detail Refinement
        self.idr = nn.ModuleList()
        for i in range(level):
            self.idr.append(
                nn.Sequential(
                    nn.Conv2d(self.fpn_time, self.idr_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(4, self.idr_dim),
                    nn.GELU(),
                    nn.Conv2d(self.idr_dim, self.fpn_time, kernel_size=3, padding=1),
                )
            )

        self.init_weight()

    def forward(
        self, data: torch.Tensor, wavelet: WaveletTransform
    ) -> tuple[torch.Tensor, WaveletCoeffDict]:

        if self.fpn is None:
            raise RuntimeError("FPN is not initialized, please run dummy_run() first.")

        coeffs = wavelet.transform(data)

        batch, time, h ,w = coeffs["A"].shape

        level = wavelet.level

        details = []

        for i in range(level):
            detail = coeffs[f"D{i + 1}"]
            b, t, c, h, w = detail.shape
            # shape: (B, T, 3, H, W)
            detail = rearrange(detail, "b t c h w -> b c h w t")

            detail = self.temporal_mlp_before(detail)

            detail = rearrange(detail, "b c h w t -> (b c) t h w", c=3, b=b, h=h, w=w)

            details.append(detail.contiguous())

        # FPN module
        fpn_output = self.fpn(details)

        for i in range(level):
            fpn_out = fpn_output[i]

            # skip connection
            fpn_out = fpn_out + self.idr[i](details[i])

            bc, t, h, w = fpn_out.shape

            # temporal MLP
            fpn_out = rearrange(fpn_out, "b t h w -> b h w t").contiguous()
            fpn_out = self.temporal_mlp_after(fpn_out)
            fpn_out = rearrange(
                fpn_out, "(b c) h w t -> b t c h w", c=3, h=h, w=w, b=batch
            )

            coeffs[f"D{i + 1}"] = fpn_out.contiguous()
        
        last_frame = coeffs["A"][:, -1, :, :]
        coeffs["A"] = repeat(last_frame, "b h w -> b t h w", t=time)
        

        reconstructed = wavelet.reverse(coeffs)

        return reconstructed.contiguous(), coeffs
