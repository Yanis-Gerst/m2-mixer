# coding=utf-8
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
import os

from multiprocessing import Pool, Lock, Manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wav_to_spectrogram(audio, f_length, t_length, idx_cnt, output, sr=8000):

    y = audio

    min_seg_length = int(np.ceil(len(y) / t_length))
    time_seg_length = min_seg_length
    noverlap = 0
    flag = False
    for i in range(min_seg_length - 1, len(y)):
        for j in range(i):
            if 113 * i - 112 * j > len(y) >= 112 * i - 111 * j:
                noverlap = j
                time_seg_length = i
                flag = True
                break
        if flag:
            break

    nfft = (f_length - 1) * 2 + 1

    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    pxx, freqs, bins, _ = ax.specgram(x=audio,
                                      NFFT=time_seg_length, pad_to=nfft, noverlap=noverlap, Fs=sr,
                                      cmap='Greys')

    fig.canvas.draw()
    size_inches = fig.get_size_inches()
    dpi = fig.get_dpi()
    width, height = fig.get_size_inches() * fig.get_dpi()
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # print("MPLImage Shape: ", np.shape(mplimage))
    imarray = np.reshape(mplimage, (int(height), int(width), 3))
    plt.close(fig)

    # if len(f) != f_length or len(t) != t_length:
    if len(bins) != t_length and len(freqs) != f_length:
        print('fucked')
        exit(1)

    with lck:
        # choose one channel of greys
        print(idx_cnt)
        output['data'].append(imarray[:, :, 0])
        idx_cnt.value += 1


def pool_init(l):
    global lck
    lck = l


def generate_melspectrogram_112x112(samples_np, output, idx_cnt,
                                    sample_rate_hz=8000,
                                    nfft_window_size=1024,
                                    fft_points=1024,
                                    hop_samples_overlap=512
                                    ):
    if not isinstance(samples_np, np.ndarray):
        raise TypeError("Input samples must be a NumPy array.")
    if samples_np.ndim != 1:
        raise ValueError("Input samples must be a 1D array.")
    if fft_points < nfft_window_size:
        fft_points = nfft_window_size

    samples_tensor = torch.from_numpy(samples_np).float()
    if samples_tensor.ndim == 1:
        samples_tensor = samples_tensor.unsqueeze(0)

    win_length_actual = nfft_window_size
    hop_length_actual = nfft_window_size - hop_samples_overlap
    n_mels_target = 112
    width_target = 112

    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate_hz,
        n_fft=fft_points,
        win_length=win_length_actual,
        hop_length=hop_length_actual,
        n_mels=n_mels_target,
        power=2.0
    )

    mel_spec = mel_spectrogram_transform(samples_tensor)

    amplitude_to_db_transform = T.AmplitudeToDB(stype='power', top_db=80)
    mel_spec_db = amplitude_to_db_transform(mel_spec)

    mel_spec_db_batched = mel_spec_db.unsqueeze(0)

    mel_spec_resized_db_tensor = F.interpolate(
        mel_spec_db_batched,
        size=(n_mels_target, width_target),
        mode='bilinear',
        align_corners=False,
    )

    mel_spec_resized_db_tensor = mel_spec_resized_db_tensor.squeeze(0)

    min_val = mel_spec_resized_db_tensor.min()
    max_val = mel_spec_resized_db_tensor.max()
    if max_val > min_val:
        mel_spec_normalized_tensor = (
            mel_spec_resized_db_tensor - min_val) / (max_val - min_val)
    else:
        mel_spec_normalized_tensor = torch.zeros_like(
            mel_spec_resized_db_tensor)

    spectrogram_image_np = mel_spec_normalized_tensor.squeeze(0).cpu().numpy()

    output['data'].append(spectrogram_image_np)
    # with lck:
    #     # choose one channel of greys
    #     print(idx_cnt)
    #     output['data'].append(spectrogram_image_np)
    #     idx_cnt.value += 1

    return spectrogram_image_np, mel_spec_resized_db_tensor.squeeze(0).cpu()


def dir_to_spectrogram(input_path, sav_path, num_processes, f_length, t_length):

    m = Manager()
    l = m.Lock()
    cnt = m.Value('int', 0)
    audio_spectrogram = m.dict({'data': m.list()})

    data = np.load(input_path)
    print(data.shape)
    # pool = Pool(processes=num_processes, initializer=pool_init, initargs=(l,))
    # for i in range(len(data)):
    #     pool.apply_async(generate_melspectrogram_112x112,
    #                      args=(data[i], audio_spectrogram, cnt))
    for i in tqdm(range(len(data))):
        generate_melspectrogram_112x112(data[i], audio_spectrogram, cnt)

    # pool.close()

    # pool.join()

    data = np.array(audio_spectrogram['data'])

    print(data.shape)
    np.save(sav_path, data)
    print("data saved at ", sav_path)


def audio_gen(audio_paths, saving_dir):
    # note that the size of the spectrogram array is not (112, 112),
    # but the spectrum size is (112, 112).
    # read the doc of plt.specgram()
    for audio_path, sav_path in zip(audio_paths, saving_dir):

        dir_to_spectrogram(audio_path, sav_path,
                           num_processes=32, f_length=112, t_length=112)


if __name__ == '__main__':
    audio_paths = ["../data/avmnist_pure/audio/train_data.npy",
                   "../data/avmnist_pure/audio/test_data.npy"]

    saving_paths = [os.path.join(os.path.dirname(
        path), "train_spec.npy" if "train" in os.path.basename(path) else "test_spec.npy") for path in audio_paths]
    audio_gen(audio_paths, saving_paths)

    print('ok')
