import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from model.whisper import Whisper
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, List


class WhisperTrainer(pl.LightningModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize Whisper model
        self.model = Whisper(
            vocab_size=cfg.model.whisper.params.decoder_block.vocab_size,
            encoder_input_dim=cfg.model.whisper.params.encoder_block.input_dim,
            embed_dim=cfg.model.whisper.params.encoder_block.embed_dim,
            num_encoder_layers=cfg.model.whisper.params.encoder_block.num_layers,
            num_decoder_layers=cfg.model.whisper.params.decoder_block.num_layers,
            num_heads=cfg.model.whisper.params.decoder_block.num_heads,
            max_source_len=cfg.model.whisper.params.encoder_block.max_len,
            max_target_len=cfg.model.whisper.params.decoder_block.max_len,
            pad_token_id=cfg.text_preprocessing.pad_token_id,
        )

        self.vocab_size=cfg.model.whisper.params.decoder_block.vocab_size

        # Loss function
        self.pad_token_id = cfg.text_preprocessing.pad_token_id
        self.label_smoothing = cfg.model.whisper.params.label_smoothing

        # Training hyperparameters
        self.learning_rate = cfg.model.whisper.hyper_params.learning_rate
        self.weight_decay = cfg.model.whisper.hyper_params.weigth_decay
        self.max_steps = cfg.model.whisper.hyper_params.max_steps
        self.adam_b1 = cfg.model.whisper.hyper_params.b1
        self.adam_b2 = cfg.model.whisper.hyper_params.b2


        # Loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()


    def forward(self, mel, tokens):
        return self.model(mel, tokens)
    
    def _compute_loss(self, logits, labels):
        """Compute cross-entropy loss with label smoothing"""

        lprobs = F.log_softmax(logits, dim=-1)

        ################### CHECKING THE PREDICTION ####################
        print(torch.argmax(lprobs, dim=-1)[0])
        print(labels[0])
        #################################################################

        lprobs = lprobs.view(-1, self.vocab_size)
        labels = labels.view(-1)

        ################ NEED TO MASK THE TENSOR FOR TARGET ################
        # # Create padding mask if needed
        # if padding_mask is not None:
        #     padding_mask = padding_mask[:, 1:]                  # Remove first token
        #     padding_mask = padding_mask.reshape(-1)
        #     non_pad_mask = ~padding_mask
        #     lprobs = lprobs[non_pad_mask]
        #     labels = labels[non_pad_mask]
        ####################################################################


        loss = self.loss_fn(lprobs, labels) 

        return loss
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        mel, tokens = batch['audio'], batch['label']
        logits, _ = self.model(mel , tokens)
        loss = self._compute_loss(logits, tokens) #  _ = padding_masks['tokens_padding_mask'])

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        mel, tokens = batch['audio'], batch['label']
        logits, _ = self.model(mel, tokens)
        loss = self._compute_loss(logits, tokens) # _ = padding_masks['tokens_padding_mask'])

        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Create optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.adam_b1, self.adam_b2)
        )

        # Create learning rate scheduler 
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=self.max_steps,
            eta_min=self.learning_rate * 0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "internval": "step",
                "frequency": 1
            }
        }

