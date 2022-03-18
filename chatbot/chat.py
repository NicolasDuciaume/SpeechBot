# This code is based on the Inference section from 
# https://towardsdatascience.com/how-to-build-your-own-chatbot-using-deep-learning-bb41f970e281
# This module allows the user to chat with the train chatbot model

import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama 
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

with open("intents.json") as file:
    data = json.load(file)

    # load trained model
    model = keras.models.load_model('chat_model')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20


def chat():
    while True:
        print(Fore.CYAN + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break
        else:
            generate_response(inp)


def generate_response(inp):
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                                                      truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            print(Fore.YELLOW + "ChatBot:" + Style.RESET_ALL, np.random.choice(i['responses']))


if __name__ == "__main__":
    print(Fore.GREEN + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
    chat()