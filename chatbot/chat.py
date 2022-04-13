# This code is based on the Inference section from
# https://towardsdatascience.com/how-to-build-your-own-chatbot-using-deep-learning-bb41f970e281
# This module allows the user to chat with the train chatbot model
# check_all_messages() and message_probability() adapted from https://github.com/federicocotogno/text_recogniton_chat/blob/master/main.py

import json
import os.path

import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import random
import pickle

file_dir = "./chatbot"
names = ["christopher", "nicolas", "nazifa", "mohammad"]
curr_user = " "
address_user_by_name = ["user name", "_default welcome", "general - bye", "greetings - nice to meet you"]

with open(os.path.join(file_dir, "intents.json")) as file:
    data = json.load(file)

    # load trained model
    model = keras.models.load_model(os.path.join(file_dir, "chat_model"))

    # load tokenizer object
    with open(os.path.join(file_dir, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open(os.path.join(file_dir, 'label_encoder.pickle'), 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20


def chat():
    while True:
        inp = input("User: ")
        if inp.lower() == "quit":
            break
        else:
            generate_response(inp)


def generate_response(inp):
    inp.lower()
    for n in names:
        if n in inp:
            global curr_user
            curr_user += n

    # encoding the input and retrieving the list of encoded tags with the associated prediction
    # the higher the prediction value, the more likely that the tag will hold the appropriate response
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                                                      truncating='post', maxlen=max_len))

    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    # use the tag generated with keras to search for the best response
    for i in data['intents']:
        if i['tag'] == tag:
            output = np.random.choice(i['responses'])
            if tag in address_user_by_name:
                output += curr_user  # using the user's name (if given) when possible
            print("ChatBot: " + output.lower())
            output.lower()
            break

    return output


if __name__ == "__main__":
    print("Start messaging with the bot (type quit to stop)!")
    chat()
