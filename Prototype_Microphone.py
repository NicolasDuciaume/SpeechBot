from ctypes import sizeof
from keras.models import load_model
import numpy as np
model=load_model('D:/SYSC4705/SpeechBot/best_model.hdf5')

all_label = ["abandon", "a", "ability", "above", "able", "abortion", "about", "abroad","absence","absolute","absolutely","absorb","abuse","academic","accept","access","accident","accompany","accomplish","according","account","accurate"]

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
duration = 1.5 # seconds
filename = 'D:/speech2/'

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



print("start")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
    channels=1, blocking=True)
    
sd.wait()
sf.write(filename+"oldSong.wav", mydata, samplerate)
print("end")
#time.sleep(0)

samples, sample_rate = librosa.load(filename+'oldSong.wav', sr = 16000)
samples = librosa.resample(samples, sample_rate * duration, 8000)
ipd.Audio(samples,rate=8000)
print(predict(samples))

os.listdir('D:/speech2')
filepath='D:/speech2'

###
#t1 = 0
#t2 = 500

#from pydub import AudioSegment 

#predicts = []

#for y in range(89):
    #newAudio = AudioSegment.from_wav(filename+"oldSong.wav")
    #newAudio = newAudio[t1:t2]
    #newAudio.export(filename+'newSong.wav', format="wav")
    #samples, sample_rate = librosa.load(filename+'newSong.wav', sr = 16000)
    #samples = librosa.resample(samples, sample_rate, 8000)
    #ipd.Audio(samples,rate=8000)
    #predicts.append(predict(samples))
   # print(predict(samples))
    #t1 = t1 * 10 #Works in milliseconds
    #t2 = t2 * 10
    
#current = ""
#occurence = 0
#real_words = ["test"]

#for y in range(len(predicts)):
#    if current == predicts[y]:
#        occurence = occurence + 1
#        if occurence == 5:
#            if real_words[len(real_words) - 1] != predicts[y]:
#                real_words.append(current)
#    else:
#        current == predicts[y]


#print(real_words)
