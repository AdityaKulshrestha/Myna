## Data Preparation

Purpose: Prepare speech-text paired datasets for training.


### Data Collection

Datasets: LibriSpeech, Common Voice, TIMIT, or custom recordings.

Requirements: Audio (16kHz, WAV format) + accurate transcripts.

- Data Cleaning

Remove background noise (e.g., noisereduce, Audacity).

- Trim leading/trailing silences (VAD tools like webrtcvad).

Text Normalization

- Lowercase, remove punctuation, expand abbreviations ("mr." → "mister").

Audio Augmentation

- Speed perturbation (±10%), noise injection, pitch shifting, room impulse responses.


#### Feature Extraction in different models

| Model                     | Input Format              | Feature Extraction Process                      | Parameters                                      | Additional Techniques                            |
|---------------------------|--------------------------|------------------------------------------------|------------------------------------------------|-------------------------------------------------|
| **Wav2Vec 2.0**           | Raw waveform             | 7-layer CNN encoder                            | Sampling rate: 16 kHz¹                         | Self-supervised pretraining with contrastive loss |
| **Whisper**               | Log-Mel spectrogram      | STFT with overlapping windows, Mel filterbanks, log transform | 80 Mel bins, n_fft: 400, hop_length: 160² | Dynamic time padding, SpecAugment masking       |
| **Conformer**             | Log-Mel spectrogram      | Strided convolutions for subsampling           | Typical: 80 Mel bins, 25ms frame length, 10ms frame shift | Multi-head self-attention, convolution modules |
| **Traditional ASR (e.g., Kaldi)** | MFCC or filterbank features | STFT, Mel filterbanks, (optional) DCT for MFCC | Typical: 13-40 MFCC coefficients or 40-80 filterbank bins³ | Delta and delta-delta features                  |
| **Wav2Vec2FeatureExtractor** | Raw waveform             | Direct feature extraction from waveform        | feature_size: 1, sampling_rate: 16000, do_normalize: True⁴¹ | Zero-mean unit-variance normalization (optional) |
| **WhisperFeatureExtractor**  | Raw waveform to spectrogram | Log-Mel spectrogram computation               | n_fft: 400, hop_length: 160, feature_size: 80⁵  | Clamping and normalization of spectrogram       |



### Text Tokenization
| Model      | Tokenizer Type                           |
|-----------|-----------------------------------------|
| **Conformer** | SentencePiece (typically BPE)       |
| **Whisper**   | Byte-Pair Encoding (BPE)           |
| **Wav2Vec2**  | Custom CTC tokenizer (Wav2Vec2CTCTokenizer) |



### Padding strategy

# Automatic Speech Recognition Models: Padding Strategies in Text Tokenization and Audio Preprocessing  

Automatic Speech Recognition (ASR) models employ distinct padding strategies during text tokenization and audio preprocessing to handle variable-length inputs. Below is a systematic analysis of these practices across three prominent architectures: Wav2Vec 2.0, Whisper, and Conformer.  

## Tabulated Comparison of Padding Strategies  

| Model      | Text Tokenization Padding                                         | Audio Preprocessing Padding                                   | Notes |
|-----------|-------------------------------------------------------------------|-------------------------------------------------------------|------------------------------------------------|
| **Wav2Vec 2.0** | Uses `<pad>` token for CTC loss alignment  | Pad/crop audio to fixed duration (e.g., 1s chunks)  | Implements attention masks for variable lengths  |
|           | Dynamic batch padding via `DataCollatorCTCWithPadding`  | Feature extractor applies zero-padding to match `MAX_SEQ_LENGTH`  | Sampling rate fixed at 16kHz  |
| **Whisper**  | Special tokens (`<startoftranscript>`, `<endoftranscript>`, etc.)  | 30s window processing with truncation/padding  | Optimized for long-form audio transcription  |
| **Conformer** | Subword tokenization with dynamic batch padding  | Dynamic chunk training for streaming  | Optimized for streaming ASR via chunk-wise processing  |
|           | Padding masks for self-attention layers  | ConvSubsampling reduces sequence length before padding  | Uses relative positional encoding for variable lengths  |

**Summary of Key Findings**
- Text Tokenization
    - All models use padding tokens (<pad>) for batch alignment
    - Whisper avoids explicit padding through decoder prompt engineering
- Audio Processing
    - Fixed-duration vs. chunked approaches create latency/accuracy tradeoffs
    - 16kHz sampling rate remains standard across architectures
- Implementation
    - Dynamic padding strategies are critical for memory efficiency
    - Positional encoding variants handle padded sequences differently


### Why do we pass the tokens as well in the Decoder block of Whisper
In Whisper (and transformer-based sequence-to-sequence models in general), tokens are passed to the decoder block because the decoder is responsible for autoregressive text generation based on both:

The previously generated tokens (or ground truth during training)
The encoded representation of the input audio
Why Pass Tokens to the Decoder?
The Whisper decoder follows a transformer-based sequence-to-sequence approach where:

The encoder processes the log-Mel spectrogram and outputs a contextual representation of the audio.
The decoder takes:
Previously generated tokens (or ground truth tokens during training)
The encoder’s output (as cross-attention input)
Positional encodings to retain order information
Using these, the decoder predicts the next token in the sequence autoregressively.

How It Works in Code
Tokens are embedded using self.token_embedding(tokens).
Positional encoding is added to keep track of sequence order.
Transformer decoder layers take these embeddings and attend to the encoder output (from the log-Mel spectrogram).
The final projection layer (self.fc) maps the decoder output to vocabulary logits.
Why Not Use Only the Encoder Output?
The encoder only extracts features from the audio, but to generate meaningful text, the model must condition predictions on past tokens. Otherwise, the model wouldn’t know what text structure or words to output next.

During training, we pass the ground truth tokens (teacher forcing).
During inference, we pass the previously generated tokens iteratively to predict the next word.

This setup allows Whisper to generate transcriptions word by word while considering both the audio features and previously generated text. 🚀


### How the first token is passed

Whisper (like many sequence-to-sequence models) uses a predefined start token (<|startoftranscript|>) to initialize the decoder.

**Simulate an empty sequence for the first token**
start_token = torch.tensor([[whisper_start_token_id]])  # Shape: [1, 1]

