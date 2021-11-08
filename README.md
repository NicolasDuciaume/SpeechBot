# SpeechBot
change batch size to be lower then dataset size
X_audio, Y_word = np.zeros((max_data*3, testParts.shape[0], testParts.shape[1], testParts.shape[2])), np.zeros((max_data*3, 3))
set 3 to be the number or words in dataset
set max_data to number of dataset