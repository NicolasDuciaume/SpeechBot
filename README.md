# SpeechBot
## Overview
This is a 4th Year Project on the topic of Text to Speech, Speech to Text and Chatbot Applications. This project involves creating a Chatbot that will use Text to Speech and Speech to Text to make it so a user can communicate with the Chatbot with speech alone.

## Contributors
- Chris D'Silva: chrisdsilva@cmail.carleton.ca
- Mohammad Abou Shaaban: mohammadabushaban@cmail.carleton.ca
- Nazifa Tanzim: nazifatanzim@cmail.carleton.ca
- Nicolas Duciaume: nicolasduciaume@cmail.carleton.ca

## Installation
### clone and cd into repo and install required packages
pip install -r requirements.txt

## Run Speechbot
cd ui_updated
python Ui.py # run Speechbot


## Run Chabot
cd chatbot
python train_chatbot.py # OPTIONAL - train the chatbot
python chat.py # run

## Run Speech-to-Text
cd STT
python Training_model.py # OPTIONAL - train the chatbot
python Prototype_Microphone.py # run