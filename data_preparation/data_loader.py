


import torch
from torch.utils.data import Dataset, DataLoader 
import librosa 
import numpy as np
from datasets import Dataset as HFDataset
from datasets import load_dataset  
import torchaudio.transforms as T
import torch.nn.functional as F
from transformers import AutoTokenizer
import hydra 
from omegaconf import DictConfig, OmegaConf


class AudioDataset(Dataset):
    def __init__(self, cfg: DictConfig, split='train'):
        """
        Args:
            dataset: The dataset object
            cfg: Hydra config containing preprocessing parameters.
            split: The dataset split (train/val/test)
        """

        self.dataset = load_dataset(cfg.dataset.name, split=split)

        # Load the dataset config
        self.audio_column = cfg.dataset.audio_col
        self.text_column = cfg.dataset.text_col

        # Load preprocessing parameters from Hydra config
        self.sample_rate = cfg.audio_preprocessing.sample_rate
        self.n_fft = cfg.audio_preprocessing.n_fft
        self.hop_length = cfg.audio_preprocessing.hop_length
        self.n_mels = cfg.audio_preprocessing.n_mels
        self.transformation_type = cfg.audio_preprocessing.transformation_type
        self.transform = cfg.audio_preprocessing.transform

        # Load text preprocessing parameters from Hydra config
        self.truncation = cfg.text_preprocessing.truncation
        self.max_padding = cfg.text_preprocessing.max_padding

        # Create a Mel spectogram transform if using torchaudio
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        # Text Tokenizer 
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            cfg.text_preprocessing.tokenizer
        )

        self.max_target_frame = cfg.audio_preprocessing.max_len

        if not self.text_tokenizer.pad_token:
            self.text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Setting the padding side to right
        self.text_tokenizer.padding_side = 'right'

    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):
        # Load the audio file (expected path structure from the dataset)
        audio_path = self.dataset[idx][self.audio_column]['array']
        audio = audio_path
        # audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Apply transformation
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        # Output shape: [n_mels, time_frames]
        mel_spec = self.mel_transform(audio_tensor)
        mel_spec_db = T.AmplitudeToDB()(mel_spec)

        current_frames = mel_spec_db.shape[-1]

        # Padd the mel spectrogram
        if current_frames < self.max_target_frame:
            pad_amount = self.max_target_frame - current_frames
            mel_spec_db = F.pad(mel_spec_db, (0, pad_amount))

        text = self.dataset[idx][self.text_column] + self.text_tokenizer.eos_token


        tokenized_transcription = self.text_tokenizer(
            text, 
            truncation=self.truncation, 
            padding='max_length',
            max_length=self.max_padding,
            return_tensors='pt'
        )['input_ids'][0]
        sample = {
            "audio": mel_spec_db,
            "label": tokenized_transcription
        }

        return sample
