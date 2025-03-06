# Democratizing AI & Audios for everyone

This repository is aimed to provide an indepth training experiences and audio analysis for everyone to understand audio processing in AI. The sole purpose of this repo is to allow everyone to get started with their own audio based models.


## Running
For testing only

- Enter the root folder (ASR)
- Run python3 -m train.train

To change any parameter from the hydra config:
```python -m train.train dataset.name=SPRINGLab/IndicVoices-R_Hindi dataset.text_col=text audio_preprocessing.max_len=50000 model.whisper.params.encoder_block.max_len=100000```



## Progress
- [x] Writter the skeleton of the framework
- [x] Dataloader using huggingface dataset and Torchaudio transformation
- [x] Training script skeleton
- [x] Integration of Hydra for easy configuration
- [ ] Readme 
- [ ] Packaged the framework