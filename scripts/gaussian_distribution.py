from scipy import stats
import librosa
import numpy as np

if __name__ == '__main__':
    path = '../data/clean_testset_wav/p232_001.wav'
    data, _ = librosa.load(path, sr=16000)
    mean = data.mean()
    std = data.std()
    D, P = stats.kstest(data, 'norm', (mean, std))  # 0.2308
    data_mag, data_phase = librosa.magphase(librosa.stft(data, n_fft=320, hop_length=160, win_length=320))
    data_mag = data_mag.flatten()
    mean = data_mag.mean()
    std = data_mag.std()
    D, P = stats.kstest(data_mag, 'norm', (mean, std))  # 0.4095  mag
