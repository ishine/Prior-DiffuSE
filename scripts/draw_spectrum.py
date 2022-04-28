import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np


def plot_specgram(wave, title="Spectrogram"):
    c = np.sqrt(np.sum((wave ** 2)) / len(wave))
    feat_wav = wave / c
    feat_wav = torch.FloatTensor(feat_wav)
    feat_x = torch.stft(feat_wav,
                        n_fft=320,
                        hop_length=160,
                        win_length=320,
                        window=torch.hann_window(320)).permute(2, 1, 0)
    feat_phase_x = torch.atan2(feat_x[-1, :, :], feat_x[0, :, :])
    feat_mag_x = torch.norm(feat_x, dim=0)
    feat_mag_x = feat_mag_x ** 0.5
    feat_x = torch.stack(
        (feat_mag_x * torch.cos(feat_phase_x), feat_mag_x * torch.sin(feat_phase_x)),
        dim=0).numpy()
    print(np.max(feat_x[0]))
    plt.hist(feat_x[0] / 3, bins=100)
    # plt.matshow(feat_x[0])
    # plt.colorbar()
    plt.show()


def plot_wav(wave, title='Wav', path=None):
    c = np.sqrt(np.sum((wave ** 2)) / len(wave))
    feat_wav = wave / c
    plt.figure()
    librosa.display.waveshow(feat_wav, sr=16000, color='black')
    plt.axis('off')
    if path:
        plt.savefig(f'../assets/{path}', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    sample_wav_path = '../data/clean_testset_wav/p232_001.wav'
    clean_wave, _ = librosa.load(sample_wav_path)
    plot_wav(clean_wave, path='clean_wav.pdf')

    sample_wav_path = '../data/noisy_testset_wav/p232_001.wav'
    noisy_wave, _ = librosa.load(sample_wav_path)
    plot_wav(noisy_wave, path='noisy_wav.pdf')
