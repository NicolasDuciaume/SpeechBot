from ctypes import sizeof
from keras.models import load_model
import numpy as np
model=load_model('./STT/best_model.hdf5')
import sounddevice as sd
import soundfile as sf
import librosa
import os
import IPython.display as ipd
import numpy as np
import time

filename = './STT/output/'
filename_full_recording = "./STT/input/"

all_label = ["abandon", "a", "ability", "above", "able", "abortion", "about", "abroad","absence","absolute","absolutely","absorb","abuse","academic","accept","access","accident","accompany","accomplish","according","account","accurate"]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)


def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]


x = 5


for z in range(0,x):
    samples, sample_rate = librosa.load(filename+str(z) + ".wav", sr = 16000)
    if((librosa.get_duration(y=samples, sr=sample_rate)) <= 1.5):
        samples = librosa.resample(samples, sample_rate * 1.5, 8000)
        ipd.Audio(samples,rate=8000)
        print(predict(samples))


