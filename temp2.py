import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy.io.wavfile import write
paths = ["../data/avmnist_pure/audio/train_spec.npy",
         "../data/avmnist_pixel_noise_hardest/audio/train_spec.npy"]

for idx, path in enumerate(paths):
    data = np.load(path)
    print(data.shape)
    plt.imshow(data[0], cmap="gray")
    plt.savefig(f"test{idx}.png")


def audio_test(path):
    data = np.load(path)
    print(data.shape)
    write("audio_test.wav", 8000, data[0])


audio_test("../data/avmnist_pure/audio/train_data.npy")
