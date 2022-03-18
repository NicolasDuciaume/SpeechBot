import pyttsx3

##Vars
engine = pyttsx3.init()
rate = engine.getProperty('rate')
volume = engine.getProperty('volume')
engine.setProperty('volume', volume-0.25)
engine.setProperty('rate', rate-75)

## Input through console
def input_console():
    input = input("Input your text that needs to be converted:\n")

## Input through text file
def input_txt_file():
    input_directory = input("Input the file path to the text file\n")
    with open(input_directory) as f:
        contents = f.read()
    engine.save_to_file(contents, "test.mp3")
    engine.runAndWait()
    print("Your audio file has been saved!")


from pydub import AudioSegment
from pydub.playback import play

def input_txt(x):
    contents = x
    engine.save_to_file(contents, "test.mp3")
    engine.runAndWait()
    print("Your audio file has been saved!")
    song = AudioSegment.from_wav("test.mp3")
    play(song)

# input_console()
# input_txt_file()

#input_txt("Hello Chris")