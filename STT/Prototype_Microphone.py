from ctypes import sizeof
from keras.models import load_model
import numpy as np
import os
file_dir = "C:\\Users\\anwar_tmk\\Documents\\Carleton\\4th Year\\4th year project\\SpeechBot\\STT\\"
model=load_model(os.path.join(file_dir, 'best_model.hdf5'))

all_label = ["hello", "hi", "hey", "would", "with", "to", "this", "on", "is", "it", "no","yes", "your", "you", "that", "not", "Nicolas", "Nazifa", "in", "I", "for", "do", "but", "as", "again", "afternoon", "and", "age", "a", "able", "about", "by","have" ]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)


def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    max = np.argmax(prob)
    return classes[index] + " " + str(max)

import sounddevice as sd
import soundfile as sf
import librosa
import IPython.display as ipd
import numpy as np
import time

samplerate = 16000  
duration = 10 # seconds

filename = os.path.join(file_dir, 'output')
filename_full_recording = os.path.join(file_dir, "input")


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
    
    
def record_and_predict():
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
        min_silence_len=300,

        # consider it silent if quieter than -16 dBFS
        silence_thresh=-30
    )

    x = 0

    for i, chunk in enumerate(audio_chunks):
        out_file = filename+ str(i) +".wav"
        chunk.export(out_file, format="wav")
        samples, sample_rate = librosa.load(filename+ str(i) +".wav", sr = 16000)
        updated_samples = pad_audio(samples, int(sample_rate * 0.75), 0.75);
        sf.write(out_file, updated_samples, sample_rate)
        samples, sample_rate = librosa.load(filename+ str(i) +".wav", sr = 16000)
        time_add = 1.5 - librosa.get_duration(y=samples, sr=sample_rate)
        time_add = time_add * 1000
        if time_add >= 0:
            silence = AudioSegment.silent(duration=time_add)
            audio = AudioSegment.from_wav(out_file)
            padded = audio + silence
            padded.export(out_file, format='wav')
            x = x + 1
   
    time.sleep(2)
    
    text = ""

    for z in range(0,x):
        samples, sample_rate = librosa.load(filename+str(z) + ".wav", sr = 16000)
        if((librosa.get_duration(y=samples, sr=sample_rate)) <= 1.5):
            samples = librosa.resample(samples, sample_rate * 1.5, 8000)
            ipd.Audio(samples,rate=8000)
            text = text + predict(samples) + " "
    
    return text


def different_rec():
    import speech_recognition as sr
    
    r = sr.Recognizer()
    
    with sr.AudioFile(filename_full_recording+"oldSong.wav") as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data)
        return text
    

def Combination_Predict():
    text1 = record_and_predict()
    text2 = different_rec()
    x = text1.split()
    y = text2.split()
    words = []
    prob = []
    prediction = ""
    for z in range(0,len(x)):
        if z % 2 == 0:
            words.append(x[z])
        else:
            prob.append(x[z])
    
    for count in range(0,len(y)):
        if(len(prob) > count):
            if (int(prob[count]) >= 80) and (prob[count] is not None):
                prediction = prediction + words[count]
            else:
                prediction = prediction + " " + y[count]
        else:
            prediction = prediction + " " + y[count]
            
    return prediction



