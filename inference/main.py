import torch
import hydra
import librosa
import torchaudio.transforms as T
import torch.nn.functional as F
from omegaconf import DictConfig
from model.whisper import Whisper  # Your standard PyTorch nn.Module class


def preprocess_audio(audio_file, cfg):
        
    mel_transform = T.MelSpectrogram(
        sample_rate=cfg.audio_preprocessing.sample_rate,
        n_fft=cfg.audio_preprocessing.n_fft,
        hop_length=cfg.audio_preprocessing.hop_length,
        n_mels=cfg.audio_preprocessing.n_mels
    )

    audio, sr = librosa.load(audio_file, sr=cfg.audio_preprocessing.sample_rate)

    # Apply transformation
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    # Output shape: [n_mels, time_frames]
    mel_spec = mel_transform(audio_tensor)
    mel_spec_db = T.AmplitudeToDB()(mel_spec)

    current_frames = mel_spec_db.shape[-1]

    # Padd the mel spectrogram audio
    if current_frames < cfg.audio_preprocessing.max_len:
        pad_amount = cfg.audio_preprocessing.max_len - current_frames
        mel_spec_db = F.pad(mel_spec_db, (0, pad_amount))

    return mel_spec_db


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Path to the trained model checkpoint
    checkpoint_path = "/root/aditya/ASR/lightning_logs/version_2/checkpoints/epoch=2-step=738.ckpt"

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Extract the model state dictionary (usually stored under 'state_dict')
    state_dict = checkpoint["state_dict"]

    # Remove 'model.' prefix if it was saved under a LightningModule wrapper
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    # Load the state dictionary into your PyTorch model
    model = Whisper(
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

    model.load_state_dict(new_state_dict)

    # Move model to evaluation mode
    model.eval()


    # generate text 
    # audio_path = "/root/aditya/ASR/sample_dataset/IISc_VaaniProject_M_RJ_Nagaur_Prak61920_0317370000_RJNGMUA_123552_11646_16124.wav"
    audio_path = "/root/aditya/ASR/sample_dataset/common_voice_hi_26010684.mp3"
    mel_spec_db = preprocess_audio(audio_path, cfg)
    print(mel_spec_db)

    output = model.generate(mel_spec_db.unsqueeze(0), max_len=64, top_p=None, top_k=None)
    print(output)

if __name__ == "__main__":
    main()
