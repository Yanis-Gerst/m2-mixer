import numpy as np
from scipy.io.wavfile import write

path = "../data/avmnist_basic_noise/audio/raw_train_data.npy"
test_p = "../data/avmnist_pure/audio/test_data.npy"
data = np.load(path)
data_p = np.load(test_p)

print(data[0])
print(data_p[0])
s = 8000

write("test_test.wav", s, data_p[0])
