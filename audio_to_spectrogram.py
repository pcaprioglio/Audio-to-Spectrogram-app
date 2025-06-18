import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import glob
import os
import cv2
import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

## Add here your correct paths
raw_file_dir = r"/Users/pietrocaprioglio/Documents/GitHub/AI-audio-visual-generative/test_audio_files"
spectrogram_results_dir = r'/Users/pietrocaprioglio/Documents/GitHub/AI-audio-visual-generative/spectrogram_results'
preprocessed_spectrogram_dir = r"/Users/pietrocaprioglio/Documents/GitHub/AI-audio-visual-generative/pre_processed"


def create_spectrogram(root_dir, out_dir, n_mels=128):
    '''
    Function to conver audiofile into a spectrogram. In the out graph the frequencies are already converted into the Mel scale. 
    '''
    audio_files = glob.glob(root_dir + '/*')

    if not os.path.exists(out_dir):
            os.mkdir(out_dir)    
    
    for file in audio_files:
        file_name = file.split('/')[-1].split('.')[0]
        y, sr = librosa.load(file)

        S = librosa.feature.melspectrogram(y=y,
                                   sr=sr,
                                   n_mels=n_mels,)
        S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot the mel spectogram
        img = librosa.display.specshow(S_db_mel,
                                    x_axis='time',
                                    y_axis='log',
                                    ax=ax)
        # ax.set_title('Mel Spectogram Example', fontsize=20)    ## Comment this line if you want to remove the title
        # fig.colorbar(img, ax=ax, format=f'%0.2f')              ## Comment this line if you want to remove the axis
        ax.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
        plt.xlabel('')
        plt.ylabel('')
        fig.savefig(out_dir + '/' + file_name + '.png',   bbox_inches='tight', pad_inches=0)


def spectrogram_preprocess(root_dir, out_dir, resize_px=224):  
        if not os.path.exists(out_dir):
                os.mkdir(out_dir)  

        files = glob.glob(root_dir + '/*')
        for file in files:
            file_name = file.split('/')[-1].split('.')[0]
            img = cv2.imread(file)
            img = cv2.resize(img, (resize_px,resize_px), interpolation=cv2.INTER_AREA)
            cv2.imwrite(out_dir + '/' + file_name + '.png', img)

if __name__ == '__main__':
      create_spectrogram(root_dir=raw_file_dir, out_dir=spectrogram_results_dir)
      spectrogram_preprocess(root_dir=spectrogram_results_dir,out_dir=preprocessed_spectrogram_dir)



