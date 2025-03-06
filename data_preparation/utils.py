from datasets import load_dataset
from omegaconf import DictConfig
from .data_loader import AudioDataset
from torch.utils.data import DataLoader, Dataset

def filter_data(example):

    return example



def load_and_split_dataset(cfg: DictConfig):
    train_dataset = AudioDataset(cfg=cfg, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    # val_dataset = AudioDataset(cfg=cfg, split='val')
    # return train_dataset, val_dataset
    return train_dataloader, None

