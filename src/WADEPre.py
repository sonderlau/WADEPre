import torch
from lightning.pytorch import LightningModule
from torch.nn import functional as F

from src.Approximation import ApproximationNetwork
from src.Detail import DetailNetwork
from src.Refiner import Refiner
from utils.metrics import log_loss
from utils.wavelet_transform import WaveletTransform, WaveletCoeffDict
from utils.zncc import zncc


class WADEPre(LightningModule):
    def __init__(
        self,
        # general params
        timesteps: int,
        spatial_size: int,
        dropout_rate: float = 0.1,
        # detail network params
        detail_idr_dim: int = 32,
        detail_feature_channel: int = 64,
        detail_layer_channels: list = [64, 128, 256],
        detail_num_blocks: int = 4,
        # approximation network params
        approx_hidden_size: int = 128,
        approx_cells: int = 3,
        # refine mixer params
        refine_hidden_dim: int = 128,
        # wavelet params
        wavelet_name: str = "bior2.4",
        wavelet_level: int = 3,
        # model params
        lr: float = 1e-3,
        # loss
        loss_a_weight: float = 1.0,
        loss_a_constant_weight: float = 0.15,
        loss_a_stop_step: int = 5000,
        loss_d_weight: float = 1.0,
        loss_recon_mean_weight: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.a_weight = loss_a_weight
        
        self.a_weight_decay = (loss_a_weight - loss_a_constant_weight) / loss_a_stop_step
        self.loss_a_constant_weight = loss_a_constant_weight
        
        self.loss_a_stop_step = loss_a_stop_step
        self.d_weight = loss_d_weight
        self.loss_recon_mean_weight= loss_recon_mean_weight

        self.detail_network = DetailNetwork(
            fpn_time=timesteps,
            idr_dim=detail_idr_dim,
            feature_channel=detail_feature_channel,
            layer_channels=detail_layer_channels,
            num_blocks=(
                detail_num_blocks
                if isinstance(detail_num_blocks, (list, tuple))
                else [detail_num_blocks] * len(detail_layer_channels)
            ),
            dropout_rate=dropout_rate
        )

        self.approx_network = ApproximationNetwork(hidden_size=approx_hidden_size, 
                                                   timesteps=timesteps,
                                                   cell_numbers=approx_cells,
                                                   dropout_rate=dropout_rate)

        self.refine_mixer = Refiner(
            time_steps=timesteps,
            hidden_dim=refine_hidden_dim,
            dropout_rate=dropout_rate
        )

        self.wavelet_transform = WaveletTransform(
            wavelet=wavelet_name, level=wavelet_level, mode="reflect"
        )

        self.lr = lr

        
        self.init_model()

    def init_model(self) -> None:
        # dummy run
        
        x = torch.randn(
            2,
            self.hparams.timesteps,
            self.hparams.spatial_size,
            self.hparams.spatial_size,
        )
        self.detail_network.dummy_run(data=x, wavelet=self.wavelet_transform)
        self.approx_network.dummy_run(data=x, wavelet=self.wavelet_transform)

    def forward(self, x: torch.Tensor):
        d_reconstruction, d_coeff = self.detail_network.forward(
            x, wavelet=self.wavelet_transform
        )
        a_reconstruction, a_coeff = self.approx_network.forward(
            x, wavelet=self.wavelet_transform
        )

        ad_coeff: WaveletCoeffDict = {"A": a_coeff["A"]}

        for l in range(1, self.wavelet_transform.level + 1):
            level_key = f"D{l}"
            level_details = d_coeff[level_key]  # Tensor shaped (B, T, 3, H, W)


            if level_details.shape[2] != 3:
                raise RuntimeError(
                    f"Expected 3 detail channels at level {l}, got shape {level_details.shape}"
                )

            ad_coeff[level_key] = level_details

        ad_reconstruction = self.wavelet_transform.reverse(ad_coeff)

        refined_out = self.refine_mixer.forward(
            AD_guide=ad_reconstruction,
            A_guide=a_reconstruction,
            D_guide=d_reconstruction,
            last_frame=x[:, -1:, :, :]
        )

        return refined_out, {
            "d_rec": d_reconstruction,
            "a_rec": a_reconstruction,
            "ad_rec": ad_reconstruction,
            "refined_out": refined_out,
            "d_coeff": d_coeff,
            "a_coeff": a_coeff,
            "ad_coeff": ad_coeff,
        }

    def compute_loss(
        self, x: dict, truth: torch.Tensor, stage: str = "train"
    ) -> torch.Tensor:

        truth_wave: WaveletCoeffDict = self.wavelet_transform.transform(truth)

        if self.global_step < self.loss_a_stop_step:
            a_weight = self.a_weight - self.global_step * self.a_weight_decay
        else:
            a_weight = self.loss_a_constant_weight

        
        main_recon = F.mse_loss(x["refined_out"], truth)


        # A coeff loss
        a_coeff_loss = zncc(x["a_coeff"]["A"], truth_wave["A"])

        # D coeff loss
        d_coeff_loss = 0.0
        for l in range(1, self.wavelet_transform.level + 1):
            d_coeff_loss += F.mse_loss(x["d_coeff"][f"D{l}"], truth_wave[f"D{l}"]) * (
                1.0 / (2**l)
            )

        
        # Reconstruction mean loss
        reconstruction_mean = (x["ad_rec"] + x["a_rec"] + x["d_rec"]) / 3
        reconstruction_mean_loss = F.mse_loss(reconstruction_mean, truth)
        
        
        total_loss = (
            main_recon + a_weight * a_coeff_loss + self.d_weight * d_coeff_loss + self.loss_recon_mean_weight * reconstruction_mean_loss
        )
                
        self.log(f"{stage}/recon", main_recon, sync_dist=True, prog_bar=False)
        self.log(f"{stage}/recon_mean", reconstruction_mean_loss, sync_dist=True, prog_bar=False)
        self.log(f"{stage}/a_coeff", a_coeff_loss, sync_dist=True, prog_bar=False)
        self.log(f"{stage}/a_weight", a_weight, sync_dist=True, prog_bar=False)
        self.log(f"{stage}/d_coeff", d_coeff_loss, sync_dist=True, prog_bar=False)
        self.log(f"{stage}/loss", total_loss, sync_dist=True, prog_bar=False)

        return total_loss

    def training_step(self, batch):
        data = batch["sequence"]
        target = batch["target"]

        output, details = self.forward(data)

        loss = self.compute_loss(details, target, stage="train")
        
        log_loss(
            pred=output,
            truth=target,
            stage="trian",
            lightning_module=self
        )

        return loss

    def validation_step(self, batch):
        data = batch["sequence"]
        target = batch["target"]

        output, details = self.forward(data)

        loss = self.compute_loss(details, target, stage="val")

        log_loss(
            pred=output,
            truth=target,
            stage="val",
            lightning_module=self
        )

        return loss

    def configure_optimizers(self):
        
        
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=0.01,
            betas=(0.9, 0.995),
        )
        
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.trainer.max_epochs,
        )
        

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            },
        }

    

    
    
    def on_load_checkpoint(self, checkpoint):
        
        state_dict = checkpoint.pop("state_dict", None)
        self.setup("predict")
        
        if state_dict is not None:
            checkpoint["state_dict"] = state_dict
