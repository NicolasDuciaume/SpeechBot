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
duration = 20 # seconds
filename = 'D:/SYSC4705/SpeechBot/output'
filename_full_recording = "D:/SYSC4705/SpeechBot/input/"

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
time.sleep(0.3)
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
    channels=1, blocking=True)
sd.wait()
sf.write(filename_full_recording+"oldSong.wav", mydata, samplerate)
print("end")
    


from pydub import AudioSegment
from pydub.silence import split_on_silence

sound_file = AudioSegment.from_wav(filename_full_recording+"oldSong.wav")
audio_chunks = split_on_silence(sound_file, 
    # must be silent for at least half a second
    min_silence_len=700,

    # consider it silent if quieter than -16 dBFS
    silence_thresh=-40
)

x = 0

for i, chunk in enumerate(audio_chunks):
    out_file = filename+ str(i) +".wav"
    chunk.export(out_file, format="wav")
    samples, sample_rate = librosa.load(filename+ str(i) +".wav", sr = 16000)
    time = 1.5 - librosa.get_duration(y=samples, sr=sample_rate)
    time = time * 1000
    #updated_samples = pad_audio(samples, sample_rate, 1.5);
    #sf.write(filename+ str(i) +".wav", updated_samples, samplerate)
    silence = AudioSegment.silent(duration=time)
    audio = AudioSegment.from_wav(out_file)
    padded = audio + silence
    print(predict(padded))
    #padded.export(out_file, format='wav')
    x = x + 1




#for i in range(x):
    #samples, sample_rate = librosa.load(filename+'/split/chunk{i}.wav', sr = 16000)
    #samples = librosa.resample(samples, sample_rate * duration, 8000)
    #ipd.Audio(samples,rate=8000)
    #print("You said:" + predict(samples))
    #time.sleep(2)
