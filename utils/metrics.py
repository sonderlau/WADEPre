from typing import Dict, Tuple
from pytorch_lightning import LightningModule
import torch

from torchmetrics.image import StructuralSimilarityIndexMeasure

SEVIR_THRESHOLDS = (16, 74, 133, 160, 181, 219)

Shanghai_THRESHOLDS = (20, 30, 35, 40)


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def compute_psd(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    assert (
        pred.dim() == 4 and target.dim() == 4
    ), "Input tensors must be 4D (B, C, H, W)"

    _, _, h, w = pred.shape
    
    pred = pred.float()
    target = target.float()

    fft_pred = torch.fft.fft2(pred, dim=(-2, -1))
    fft_target = torch.fft.fft2(target, dim=(-2, -1))

    power_pred = torch.abs(fft_pred) ** 2 / (h * w)
    power_target = torch.abs(fft_target) ** 2 / (h * w)

    return torch.nn.functional.mse_loss(power_pred, power_target)


def compute_csi(
    pred: torch.Tensor, target: torch.Tensor, threshold: float
) -> torch.Tensor:
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    tp = torch.sum((pred == 1) & (target == 1))
    fp = torch.sum((pred == 1) & (target == 0))
    fn = torch.sum((pred == 0) & (target == 1))

    csi = tp / (tp + fp + fn)

    if (tp + fp + fn) > 0:
        return csi
    else:
        return torch.tensor([0.0], device=pred.device, dtype=pred.dtype)


def compute_csi_mean(pred: torch.Tensor, target: torch.Tensor, thresholds: Tuple[int]) -> torch.Tensor:

    result = []
    for threshold in thresholds:
        csi = compute_csi(pred, target, threshold)
        result.append(csi)

    return torch.tensor(result, device=pred.device).mean()


def log_loss(
    pred: torch.Tensor,
    truth: torch.Tensor,
    stage: str,
    lightning_module: LightningModule,
):
    # Root Mean Square Error
    rmse = compute_rmse(pred, truth)
    lightning_module.log(f"{stage}/rmse", rmse, sync_dist=True)

    # CSI Mean
    csi_mean = compute_csi_mean(pred * 255, truth * 255)
    lightning_module.log(f"{stage}/csi_mean", csi_mean, sync_dist=True, prog_bar=True)

    # CSI 181 Threshold
    csi_181 = compute_csi(pred * 255, truth * 255, threshold=181)
    lightning_module.log(f"{stage}/csi_181", csi_181, sync_dist=True)

    # CSI 219 Threshold
    csi_219 = compute_csi(pred * 255, truth * 255, threshold=219)
    lightning_module.log(f"{stage}/csi_219", csi_219, sync_dist=True)


def compute_loss(pred: torch.Tensor, truth: torch.Tensor,):
    # Root Mean Square Error
    rmse = compute_rmse(pred, truth)

    # Power Spectral Density
    psd = compute_psd(pred, truth)


    # CSI Mean
    csi_mean = compute_csi_mean(pred * 255, truth * 255)

    # CSI 181 Threshold
    csi_181 = compute_csi(pred * 255, truth * 255, threshold=181)

    # CSI 219 Threshold
    csi_219 = compute_csi(pred * 255, truth * 255, threshold=219)

    # SSIM
    ssim_value =  StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    ssim_value = ssim_value(pred, truth)
    return {
        "rmse": rmse,
        "psd": psd,
        "csi_mean": csi_mean,
        "csi_181": csi_181,
        "csi_219": csi_219,
        "ssim": ssim_value
    }