# This code is based on the Inference section from 
# https://towardsdatascience.com/how-to-build-your-own-chatbot-using-deep-learning-bb41f970e281
# This module allows the user to chat with the train chatbot model

import json
import os.path

import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle





def chat():
    while True:
        print(Fore.CYAN + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break
        else:
            generate_response(inp)

def modeling():
    import json 
    import numpy as np 
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.preprocessing import LabelEncoder

    with open('D:/SYSC4705/SpeechBot/chatbot/intents.json') as file:
        data = json.load(file)

    training_sentences = [] # holds training data
    training_labels = [] # holds target labels (i.e. tags)
    labels = []
    responses = []

    # iterate through the json file and store training data and tags
    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses.append(intent['responses'])
    
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
        
    num_classes = len(labels)  

    # convert labels into model understandable form
    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(training_labels)
    training_labels = lbl_encoder.transform(training_labels)

    vocab_size = 1000
    embedding_dim = 16
    max_len = 20
    oov_token = "<OOV>" # deals with out of vocabulary words

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    # makes all training test sequences the same size
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    epochs = 500
    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)
    # save the trained model
    model.save("chat_model")
    return model, tokenizer, max_len, lbl_encoder, data

model, tokenizer, max_len, lbl_encoder, data = modeling()

def generate_response(inp):
    
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                                                      truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            output = np.random.choice(i['responses'])
            print(Fore.YELLOW + "ChatBot:" + Style.RESET_ALL, output)
            return output

#if __name__ == "__main__":
    #print(Fore.GREEN + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
    #chat()
