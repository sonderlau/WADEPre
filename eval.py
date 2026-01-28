import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary,
    
)
from lightning.pytorch.loggers import CSVLogger

from src.WADEPre import WADEPre

torch.set_float32_matmul_precision("high")

MODEL_NAME = "StormWave_Eval"


def main():
    # seed
    pl.seed_everything(42, workers=True)

    # callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename=MODEL_NAME,
            monitor="val/loss",
            save_last=True,
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="step", log_weight_decay=False),
        RichProgressBar(),
        RichModelSummary(max_depth=3),
        # EarlyStopping(monitor="val/csi_mean", patience=15, mode="min"),
    ]

    # logger
    csv_logger = CSVLogger(
        save_dir="logs", name=MODEL_NAME, flush_logs_every_n_steps=10
    )

    # init model
    # StormWave(

    # )
    m = WADEPre.load_from_checkpoint("WADEPre_SEVIR.ckpt",
        timesteps=6,
        spatial_size=128,
        loss_a_stop_step=3000,
        lr=1.5e-4,
        wavelet_level=3,
        detail_layer_channels=[64, 128, 256],
        detail_num_blocks=4,
        loss_a_weight=0.1,
        loss_a_constant_weight=0.01,
        loss_d_weight=0.05,
        loss_recon_mean_weight=0.005,
        detail_idr_dim=64,
        detail_feature_channel=128,
        refine_hidden_dim=6 * 96,
        approx_hidden_size=512,
        approx_cells=3,
        dropout_rate = 0.1
    )
    

    # trainer
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu",
        devices=2,
        precision="32",
        callbacks=callbacks,
        enable_model_summary=True,
        logger=csv_logger
    )

    trainer.predict(model=m, datamodule=dataset)



if __name__ == "__main__":
    main()
