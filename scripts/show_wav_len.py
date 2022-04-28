import glob
import soundfile as sf
import matplotlib.pyplot as plt


path_list = glob.glob('../data/clean_testset_wav/*.wav')
for path in path_list:
    data, _ = sf.read(path)
    print(len(data))
    plt.hist(data)
    plt.show()