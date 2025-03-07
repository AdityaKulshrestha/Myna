import hydra
import lightning as pl
import torch
from .utils import count_parameters
from torch.utils.data import DataLoader
from data_preparation.data_loader import AudioDataset
from omegaconf import DictConfig
from .whisper_trainer import WhisperTrainer
from torch.utils.data import random_split, DataLoader
from lightning_habana.pytorch.strategies import HPUDDPStrategy


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    dataset = AudioDataset(cfg, split='train')
    train_size = int(cfg.dataset.train_split * len(dataset))
    val_size = len(dataset) - train_size

    # Split dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=cfg.training.batch,
        shuffle=False
    )

    # Load model
    model = WhisperTrainer(cfg)

    ### Model Parameters
    parameter_count = count_parameters(model)
    print("Total Number of Trainable Parameter: ", parameter_count)

    if cfg.training.strategy == "ddp":
        trainer = pl.Trainer(
            accelerator="hpu",
            devices=2,
            max_epochs=cfg.training.epochs,
            strategy=HPUDDPStrategy(parallel_devices=[torch.device('hpu')]*2, find_unused_parameters=True)     # When you pass device=2 -> It automatically selects HPUDDPStrategy
        )
    else:
        trainer = pl.Trainer(
            accelerator="hpu",
            devices=1,
            max_epochs=cfg.training.epochs
        )

    # Train model
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
