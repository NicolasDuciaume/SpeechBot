from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi


def updateCounter(tet):
    if tet == "voiceChat":
        widget.setCurrentIndex(widget.currentIndex() + 2)
    if tet == "textChat":
        widget.setCurrentIndex(widget.currentIndex() + 1)


class MainMenu(QMainWindow):
    def __init__(self):
        super(MainMenu, self).__init__()
        loadUi('MainMenu.ui', self)
        self.setWindowTitle('Main Menu')
        self.voiceChat.clicked.connect(lambda: updateCounter("voiceChat"))
        self.textChat.clicked.connect(lambda: updateCounter("textChat"))


def MainScreen():
    widget.setCurrentIndex(0)


class TextChatBot(QMainWindow):
    def __init__(self):
        super(TextChatBot, self).__init__()
        loadUi('textChatBot.ui', self)
        self.setWindowTitle('Text Chat Bot')
        self.send.clicked.connect(self.onSendCl)
        self.mainScreen.clicked.connect(MainScreen)

    def onSendCl(self):
        save = self.sendtext.text()
        self.sendtext.setText("")
        inp = input("Enter ChatBot Response: ")  # Replace with Function to generate bots response
        self.sendandrec.append('\n' + 'you: ' + save + '\n' + 'Bot: ' + inp)


class VoiceChatBot(QMainWindow):
    def __init__(self):
        super(VoiceChatBot, self).__init__()
        loadUi('voiceChatBot.ui', self)
        self.setWindowTitle('Voice Chat Bot')
        self.start.clicked.connect(self.StartPressed)
        self.end.clicked.connect(self.EndPressed)
        self.mainScreen.clicked.connect(MainScreen)

    def StartPressed(self):
        print("Start button pressed")

    def EndPressed(self):
        print("End button Pressed")
        # inp = input("Enter ChatBot Response: ")  Replace with Function to generate bots response
        # ADD RECORDED TEXT FROM THE SPEECH IN PLACE OF RECORDED_AUDIO
        # self.sendandrec.append('\n' + 'you: ' + RECORDED_AUDIO + '\n' + 'Bot: ' + inp)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    widget = QtWidgets.QStackedWidget()

    Screen1 = MainMenu()
    Screen2 = TextChatBot()
    Screen3 = VoiceChatBot()

    widget.addWidget(Screen1)
    widget.addWidget(Screen2)
    widget.addWidget(Screen3)

    widget.setFixedWidth(400)
    widget.setFixedHeight(350)
    widget.show()
    sys.exit(app.exec_())
