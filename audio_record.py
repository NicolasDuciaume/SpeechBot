import pyaudio
import wave


CHUNK = 1024
FORMAT, CHANNELS = pyaudio.paInt16, 1
RATE = 16000
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "C:/Users/anwar_tmk/Documents/Carleton/4th Year/4th year project/SpeechBot/input/audio/Christopher/Christopher53.wav"


num = 91
word = "Christopher"
for x in range(10):
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

	WAVE_OUTPUT_FILENAME = "C:/Users/anwar_tmk/Documents/Carleton/4th Year/4th year project/SpeechBot/input/audio/{0}/{1}{2}.wav".format(word, word, num)
	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()
	num += 1