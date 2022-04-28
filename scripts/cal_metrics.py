from utils.metrics import *
import glob
import soundfile as sf
import os
import librosa
from tqdm import tqdm

noisy_root = '../data/noisy_testset_wav'
clean_root = '../data/clean_testset_wav'
raw_paths = [x.split('/')[-1] for x in glob.glob(noisy_root + '/*.wav')]

all_csig_list, all_cbak_list, all_covl_list, all_pesq_list, all_ssnr_list = [], [], [], [], []

for index in tqdm(range(len(raw_paths))):
    noisy, _ = librosa.load(os.path.join(noisy_root, raw_paths[index]), sr=16000)
    clean, _ = librosa.load(os.path.join(clean_root, raw_paths[index]), sr=16000)
    result = compareone((clean, noisy))
    all_csig_list.append(result[0])
    all_cbak_list.append(result[1])
    all_covl_list.append(result[2])
    all_pesq_list.append(result[3])
    all_ssnr_list.append(result[4])
print(np.mean(all_csig_list),  # 3.35
      np.mean(all_cbak_list),  # 2.44
      np.mean(all_covl_list),  # 2.62
      np.mean(all_pesq_list),  # 1.97
      np.mean(all_ssnr_list))  # 1.67
