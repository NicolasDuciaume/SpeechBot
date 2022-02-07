import sounddevice as sd
import soundfile as sf
import librosa
import os
import IPython.display as ipd
import time

samplerate = 16000  
duration = 1.5 # seconds
filename = 'D:/SYSC4705/SpeechBot/input/audio/accurate/accurate'

for x in range(50):
    print("start")
    print(x)
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
        channels=1, blocking=True)
    print("end")
    sd.wait()
    sf.write(filename+str(x)+".wav", mydata, samplerate)