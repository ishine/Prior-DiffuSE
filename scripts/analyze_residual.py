import glob
import librosa
import librosa.display
from scipy import stats
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    raw_path = '../data/clean_testset_wav/'
    all_paths = glob.glob(raw_path+'*.wav')
    all_paths = [path.split('/')[-1] for path in all_paths]
    redirected_path = '../assets/wav/grn/'
    redirected_all_paths = [redirected_path + path for path in all_paths]
    residual_result = []
    counter = 10
    for path in tqdm(all_paths):
        clean_one, _ = librosa.load(raw_path+path, sr=16000)
        estimate_one, _ = librosa.load(redirected_path + path, sr=16000)
        residual_one = clean_one - estimate_one
        residual_result.append(residual_one[:16000])
        librosa.display.waveplot(clean_one, 16000)
        librosa.display.waveplot(estimate_one, 16000)
        librosa.display.waveplot(residual_one, 16000)
        plt.show()
        counter -= 1
        if counter < 0:
            break
        # exit()
    # mean = np.mean(residual_result, axis=-1)
    # std = np.std(residual_result, axis=-1)
