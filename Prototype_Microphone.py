from ctypes import sizeof
from keras.models import load_model
import numpy as np
model=load_model('D:/speech2/best_model.hdf5')

all_label = ["yes", "no", "up", "down","left", "right", "on", "off", "stop", "go"]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)


def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]

import sounddevice as sd
import soundfile as sf
import librosa
import os
import IPython.display as ipd
import time

samplerate = 16000  
duration = 1 # seconds
filename = 'D:/speech2/test'

def pad_audio(data, fs, T):
    # Calculate target number of samples
    N_tar = fs
    # Calculate number of zero samples to append
    shape = data.shape
    # Create the target shape    
    N_pad = N_tar - shape[0]
    #print("Padding with %s seconds of silence" % str(N_pad/fs) )
    shape = (N_pad,) + shape[1:]
    # Stack only if there is something to append    
    if shape[0] > 0:                
        if len(shape) > 1:
            return np.vstack((np.zeros(shape),
                              data))
        else:
            return np.hstack((np.zeros(shape),
                              data))
    else:
        return data

#print("start")
for x in range(6):
    print("start")
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
        channels=1, blocking=True)
    mydata = pad_audio(mydata,samplerate, 0.2)
    sd.wait()
    sf.write(filename+str(x)+".wav", mydata, samplerate)
    print("end")
    #time.sleep(0)

#print("end")

os.listdir('D:/speech2')
filepath='D:/speech2'

for y in range(6):
    samples, sample_rate = librosa.load(filepath + '/' + 'test' + str(y) + '.wav', sr = 16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    ipd.Audio(samples,rate=8000)
    print(predict(samples))
