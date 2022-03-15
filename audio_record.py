import sounddevice as sd
import soundfile as sf
import librosa
import os
import IPython.display as ipd
import numpy as np
import time

samplerate = 16000  
duration = 70 # seconds
filename = 'D:/SYSC4705/SpeechBot/input/audio/by/by'
filename_full_recording = "D:/SYSC4705/SpeechBot/input/"
#filename = './input/rerecord/yes'

def pad_audio(data, fs, T):
    # Calculate target number of samples
    N_tar = fs
    # Calculate number of zero samples to append
    shape = data.shape
    # Create the target shape    
    N_pad = N_tar - shape[0]
    print("Padding with %s seconds of silence" % str(N_pad/fs) )
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
    
   


#for x in range(50):
 #   print("\n{0}".format(x))
  #  print("start")
    
 #   mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
 #       channels=1, blocking=True)
  #  print("end")
  #  sd.wait()
  #  sf.write(filename+str(x)+".wav", mydata, samplerate)
  #  time.sleep(0.5)