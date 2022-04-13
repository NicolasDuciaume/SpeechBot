import pyttsx3
from pydub import AudioSegment
from pydub.playback import play


##Vars
engine = pyttsx3.init()
rate = engine.getProperty('rate')
volume = engine.getProperty('volume')
engine.setProperty('volume', volume-0.25)
engine.setProperty('rate', rate-75)

## Input through console
def input_console():
    text = input("Input your text that needs to be converted:\n")
    save_play(text)

## Input through text file
def input_txt_file():
    input_directory = input("Input the file path to the text file\n")
    with open(input_directory) as f:
        contents = f.read()
    save_play(contents)

## Input through method param
def input_txt(x):
    contents = x
    save_play(contents)

def save_play(contents):
    engine.save_to_file(contents, "test.mp3")
    engine.runAndWait()
    print("Your audio file has been saved!")
    audio = AudioSegment.from_wav("test.mp3")
    play(audio)



# input_console()
# input_txt_file()
# input_txt("Hi this is to test the text-to-speech functionality of the python library pyttsx3. Welcome to the 4th Year Project titled Text to Speech, Speech to Text, ChatBot applications. This project is done by Chris D'Silva, Nazifa Tanzim, Nicolas Duciaume and Mohammad Aboushaaban.")