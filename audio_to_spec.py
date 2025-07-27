import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def wav_to_spectrogram(samples, sample_rate=8000, f_length=112, t_length=112):
    """ Creates a spectrogram of a wav file.

    :param audio_dir: path of wav files
    :param file_name: file name of the wav file to process
    :param noise_path: path of noise wav file
    :return:
    """

    min_seg_length = int(np.ceil(len(samples) / t_length))
    time_seg_length = min_seg_length
    noverlap = 0
    flag = False
    for i in range(min_seg_length - 1, len(samples)):
        for j in range(i):
            if 113 * i - 112 * j > len(samples) >= 112 * i - 111 * j:
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
    pxx, freqs, bins, _ = ax.specgram(x=samples,
                                      NFFT=time_seg_length, pad_to=nfft, noverlap=noverlap, Fs=sample_rate,
                                      cmap='Greys')

    plt.close(fig)

    if len(bins) != t_length and len(freqs) != f_length:
        print('fucked')
        exit(1)

    return pxx


# def create_spec_file(audio_paths: list[str]):
#     for audio_path in audio_paths:
#         base_dir = os.path.dirname(audio_path)
#         base_filename = os.path.basename(audio_path)
#         output_filename = "train_spec.npy" if "train" in base_filename else "test_spec.npy"
#         output_path = os.path.join(base_dir, output_filename)
#         data = np.load(audio_path)
#         test = data[0]
#         if test.shape == (112, 112):
#             print("already spec")
#             exit(0)
#         new_data = []

#         for audio in tqdm(data):
#             # Ignore channel dimension of 1
#             new_data.append(wav_to_spectrogram(audio))

#         new_data = np.array(new_data)
#         print("Data saved at " + output_path)
#         np.save(output_path, new_data)


def create_spec_file(audio_paths: list[str]):
    for audio_path in audio_paths:
        base_dir = os.path.dirname(audio_path)
        base_filename = os.path.basename(audio_path)
        output_filename_base = "train_spec_baseù.npy" if "train" in base_filename else "test_spec.npy"
        output_path = os.path.join(base_dir, output_filename_base)
        temp_output_path = output_path + ".tmp"

        print(f"Processing {audio_path}...")
        try:
            data = np.load(audio_path, mmap_mode='r')
        except FileNotFoundError:
            print(f"Error: Input file not found at {audio_path}. Skipping.")
            continue
        except Exception as e:
            print(f"Error loading {audio_path}: {e}. Skipping.")
            continue

        if data.ndim == 3 and data.shape[1:] == (112, 112):
            print(
                f"Data in {audio_path} (shape: {data.shape}) already appears to be in spectrogram format. Skipping conversion.")
            if isinstance(data, np.memmap):
                del data
            continue

        if data.ndim == 0 or data.shape[0] == 0:
            print(f"Data in {audio_path} is empty or invalid. Skipping.")
            if isinstance(data, np.memmap):
                del data
            continue

        num_samples = data.shape[0]
        spec_shape = (112, 112)
        dtype_to_use = np.float32
        print(num_samples)

        try:
            output_memmap = np.memmap(
                temp_output_path, dtype=dtype_to_use, mode='w+', shape=(num_samples, 112, 112))
        except Exception as e:
            print(
                f"Error creating memory-mapped file {temp_output_path}: {e}. Skipping this input file.")
            if isinstance(data, np.memmap):
                del data
            continue

        print(
            f"Generating spectrograms for {base_filename} and saving to {output_path} (via {temp_output_path})")
        for i in tqdm(range(num_samples), desc=f"Processing {os.path.basename(audio_path)}"):
            audio_sample_data = data[i]

            spectrogram = wav_to_spectrogram(
                audio_sample_data, sample_rate=8000, f_length=112, t_length=112)
            print(spectrogram.shape)
            if i == 1:
                plt.imshow(spectrogram, cmap="gray")
                plt.savefig(f"test_spec{i}.png")
            output_memmap[i] = spectrogram

        output_memmap.flush()
        del output_memmap

        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_output_path, output_path)
            print(f"Data successfully saved at {output_path}")
        except Exception as e:
            print(
                f"Error renaming temporary file {temp_output_path} to {output_path}: {e}")
            if os.path.exists(temp_output_path):
                print(f"Temporary file {temp_output_path} still exists.")

        if isinstance(data, np.memmap):
            del data


audio_path = ["../data/avmnist_pure/audio/train_data.npy"]

create_spec_file(audio_path)
