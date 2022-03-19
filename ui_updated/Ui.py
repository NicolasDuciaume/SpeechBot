from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
import sys
import os
sys.path.insert(1, 'D:/SYSC4705/SpeechBot/')
import STT.Prototype_Microphone as PM
import TTS.TTS as tts
import chatbot.chat as cb

file_dir = "D:/SYSC4705/SpeechBot/ui_updated"


def updateCounter(tet):
    if tet == "voiceChat":
        widget.setCurrentIndex(widget.currentIndex() + 2)
    if tet == "textChat":
        widget.setCurrentIndex(widget.currentIndex() + 1)


class MainMenu(QMainWindow):
    def __init__(self):
        super(MainMenu, self).__init__()
        loadUi(os.path.join(file_dir, 'MainMenu.ui'), self)
        self.setWindowTitle('Main Menu')
        self.voiceChat.clicked.connect(app.quit)
        self.textChat.clicked.connect(lambda: updateCounter("textChat"))


def MainScreen():
    widget.setCurrentIndex(0)


class TextChatBot(QMainWindow):
    def __init__(self):
        super(TextChatBot, self).__init__()
        loadUi(os.path.join(file_dir, 'textChatBot.ui'), self)
        self.setWindowTitle('Text Chat Bot')
        self.send.clicked.connect(self.onSendCl)
        self.start.clicked.connect(self.StartPressed)
        self.mainScreen.clicked.connect(MainScreen)
        
    def StartPressed(self):
        self.sendtext.setText(PM.Combination_Predict())
    

    def onSendCl(self):
        save = self.sendtext.text()
        print("user input: {}".format(save))
        self.sendtext.setText("")
        reply = cb.generate_response(save) # send user input to chatbot and receive reply
        self.sendandrec.append('\n' + 'you: ' + save + '\n' + 'Bot: ' + reply)
        tts.input_txt(reply)


class VoiceChatBot(QMainWindow):
    def __init__(self):
        super(VoiceChatBot, self).__init__()
        loadUi(os.path.join(file_dir, 'voiceChatBot.ui'), self)
        self.setWindowTitle('Voice Chat Bot')
        self.start.clicked.connect(self.StartPressed)
        self.end.clicked.connect(self.EndPressed)
        self.mainScreen.clicked.connect(MainScreen)

    def StartPressed(self):
        print("Start button pressed")

    def EndPressed(self):
        print("End button Pressed")
        # TODO: what needs to be changes here
        # reply = cb.generate_response(user_inp)  # send user input to chatbot and receive reply
        # ADD RECORDED TEXT FROM THE SPEECH IN PLACE OF RECORDED_AUDIO
        # self.sendandrec.append('\n' + 'you: ' + RECORDED_AUDIO + '\n' + 'Bot: ' + reply)
    




if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    widget = QtWidgets.QStackedWidget()

    Screen1 = MainMenu()
    Screen2 = TextChatBot()
    # Screen3 = VoiceChatBot()

    widget.addWidget(Screen1)
    widget.addWidget(Screen2)
    # widget.addWidget(Screen3)

    widget.setFixedWidth(500)
    widget.setFixedHeight(500)
    widget.show()
    sys.exit(app.exec_())
