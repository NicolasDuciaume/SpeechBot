from ctypes import sizeof
from keras.models import load_model
import numpy as np
model=load_model('D:/speech2/best_model.hdf5')

all_label = ["abandon", "a", "ability"]

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
<<<<<<< HEAD
        audio = audio[-(audio_lenght*voice_max_length):]
    spectrogram = tf.signal.stft(audio, frame_length=1024, frame_step=frame_step)
    spectrogram = (tf.math.log(tf.abs(tf.math.real(spectrogram)))/tf.math.log(tf.constant(10, dtype=tf.float32))*20)-60
    spectrogram = tf.where(tf.math.is_nan(spectrogram), tf.zeros_like(spectrogram), spectrogram)
    spectrogram = tf.where(tf.math.is_inf(spectrogram), tf.zeros_like(spectrogram), spectrogram)
    voice_length, voice = 0, []
    nb_part = len(audio)//audio_lenght
    part_length = len(spectrogram)//nb_part
    partsCount = len(range(0, len(spectrogram)-part_length, int(part_length/2)))
    parts = np.zeros((partsCount, part_length, 513))
    for i, p in enumerate(range(0, len(spectrogram)-part_length, int(part_length/2))):
        part = spectrogram[p:p+part_length]
        parts[i] = part
    return parts

# model = tf.keras.models.load_model('D:/best_model/model_word')
model = tf.keras.models.load_model('./model_word')

CHUNK = 1024
FORMAT, CHANNELS = pyaudio.paInt16, 1
RATE = 16000
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

test_audio = audioToTensor(WAVE_OUTPUT_FILENAME)
result = model.predict(np.array([test_audio]))
max = np.argmax(result)
print("decoded_sentence: ", result, max, idToWord[max])


=======
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

<<<<<<< HEAD
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
=======
for y in range(6):
    samples, sample_rate = librosa.load(filepath + '/' + 'test' + str(y) + '.wav', sr = 16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    ipd.Audio(samples,rate=8000)
    print(predict(samples))
>>>>>>> main
>>>>>>> 5d6c62b19751d1f84ce65ece88f8fc7bcb70d8d0
