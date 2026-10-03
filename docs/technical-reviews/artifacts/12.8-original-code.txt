from tiny_perceptron.multimodal import tone, log_mel, AudioEncoder

wave = tone()[None]
spectrogram = log_mel(wave)
encoder = AudioEncoder(bands=16, width=8)
features = encoder(wave)
print("頻譜", tuple(spectrogram.shape))
print("時間特徵", tuple(features.shape))
