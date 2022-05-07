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

def plot_stft(spec, wav_len, title=None, ylabel="freq_bin", aspect="auto", xmax=None, path=None):
    # feat_x: [2, f, t]

    # plt.hist(feat_x[0] / 3, bins=100)
    # plt.matshow(feat_x[0])
    # plt.colorbar()
    esti_mag, esti_phase = torch.norm(spec, dim=0), torch.atan2(spec[-1, :, :],
                                                                  spec[0, :, :])

    esti_mag = esti_mag ** 2
    esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=0)
    tf_esti = esti_com.permute(2, 1, 0).cpu()


    fig, axs = plt.subplots(1, 2)
    axs[0].set_title(title or "Spectrogram (db)")
    axs[0].set_ylabel(ylabel)
    axs[0].set_xlabel("frame")
    im = axs[0].imshow(librosa.power_to_db(tf_esti_n[0].numpy()), origin="lower", aspect=aspect)
    im2 = axs[1].imshow(librosa.power_to_db(tf_esti_n[1].numpy()), origin="lower", aspect=aspect)
    if xmax:
        axs[0].set_xlim((0, xmax))
    fig.colorbar(im)
    plt.savefig(title+'.png',dpi=300)


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
