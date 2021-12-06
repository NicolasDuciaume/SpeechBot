import pyaudio
import wave

CHUNK = 1024
FORMAT, CHANNELS = pyaudio.paInt16, 1
RATE = 16000
RECORD_SECONDS = 2

# Change the following 3 variables (num, word, iterations) according to the intended filename and path
# num will be used as part of the filename and idicates the iteration of the word that is being recorded.
# word is used for the filename and file path. 
# for example, if you are recording the word "hello" for the first time, num = 1, word = "hello". this will be stored in a folder called
# "hello" and the filename will be "hello1.wav"
# iterations indicates the number of recordings to be made for the specific word in one go.
# if num=1, word="hello", and iterations=5, then you will be prompted to record the word 5 times in 1 session
# the expected outputs in folder "hello" will be hello1.wav, hello2.wav, hello3.wav, hello4.wav, hello5.wav
num = 105
word = "Nicolas"
iterations = 2
for x in range(iterations):
	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
					channels=CHANNELS,
					rate=RATE,
					input=True,
					frames_per_buffer=CHUNK)


	print(x)
	print("* recording")

	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)

	print("* done recording")

	stream.stop_stream()
	stream.close()
	p.terminate()

	WAVE_OUTPUT_FILENAME = "./input/audio/{0}/{1}{2}.wav".format(word, word, num)
	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()
	
	num += 1