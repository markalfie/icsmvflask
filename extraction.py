import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import python_speech_features
from IPython import display
import os
import scipy, sklearn, urllib
import timeit
from glob import glob

scale_file = 'VocalSet/FULL/female1/arpeggios/breathy/f1_arpeggios_breathy_a.wav'
print (scale_file)
filename = "f1_arpeggios_breathy_a"
    # load file

    # call log mel spectrogram
y, sr = librosa.load(scale_file)
ps = librosa.feature.melspectrogram(y=y, sr=sr)
ps_db= librosa.power_to_db(ps, ref=1.0)
librosa.display.specshow(ps_db, x_axis='s', y_axis='log')
plt.tight_layout()
plt.axis('off')
plt.savefig("static/" + filename + "_logmel.png") # CHEST/f1_arpeggios_belt_c_e.wav_logmel.png
plt.clf()

    # call log mfcc
x, fs = librosa.load(scale_file)
mfccs = librosa.feature.mfcc(x, sr=fs)
mfccs = sklearn.preprocessing.scale(mfccs)
librosa.display.specshow(mfccs, sr=fs, x_axis='time')
plt.tight_layout()
plt.axis('off')
plt.savefig("static/" + filename + "mfcc.png") 
# CHEST/f1_arpeggios_belt_c_e.wav_logmel.png