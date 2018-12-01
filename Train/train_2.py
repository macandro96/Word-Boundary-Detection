# Code Credits https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html


import keras.backend as K
from keras.layers import LSTM, Input, TimeDistributed
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
# import cv2
from tempfile import TemporaryFile
from keras.models import Model
import pickle
from keras.models import load_model
from keras.layers import Activation, dot, concatenate


def get_model():
  
  encoder_inputs = Input(shape=(None, 1))
  encoder = LSTM(100, return_sequences=True)(encoder_inputs)
  # encoder_outputs, state_h, state_c = encoder(encoder_inputs)
  
  encoder_last = encoder[:,-1,:]

  
  decoder_inputs = Input(shape=(None, 50))
  
  decoder_lstm = LSTM(100, return_sequences=True)(decoder_inputs, initial_state = [encoder_last, encoder_last])
  attention = dot([decoder_lstm, encoder], axes=[2,2])
  attention = Activation('softmax')(attention)
  context = dot([attention, encoder], axes=[2,1])

  decoder_combined_context = concatenate([context, decoder_lstm])

  output = TimeDistributed(Dense(128))(decoder_combined_context)
  output = TimeDistributed(Dense(1))(output)
  # decoder_outputs1, _, _ = decoder_lstm(decoder_inputs,
  #                                      initial_state=encoder_states)
  # decoder_dense1 = Dense(80, activation='relu')
  # decoder_outputs1 = decoder_dense1(decoder_outputs1)
  # decoder_dense = Dense(2)
  # decoder_outputs = decoder_dense(decoder_outputs1)

  # Define the model that will turn
  # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
  model = Model([encoder_inputs, decoder_inputs], output)
  model.compile(optimizer='adam', loss='mean_squared_error')
  return model
  
def get_model2():
  # Define an input sequence and process it.
  encoder_inputs = Input(shape=(None, 1))
  encoder = LSTM(100, return_state=True)
  encoder_outputs, state_h, state_c = encoder(encoder_inputs)
  # We discard `encoder_outputs` and only keep the states.
  encoder_states = [state_h, state_c]

  # Set up the decoder, using `encoder_states` as initial state.
  decoder_inputs = Input(shape=(None, 50))
  # We set up our decoder to return full output sequences,
  # and to return internal states as well. We don't use the 
  # return states in the training model, but we will use them in inference.
  decoder_lstm = LSTM(100, return_sequences=True, return_state=True)
  decoder_outputs1, _, _ = decoder_lstm(decoder_inputs,
                                       initial_state=encoder_states)
  decoder_dense1 = Dense(80, activation='relu')
  decoder_outputs1 = decoder_dense1(decoder_outputs1)
  decoder_dense = Dense(2)
  decoder_outputs = decoder_dense(decoder_outputs1)

  # Define the model that will turn
  # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
  model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
  model.compile(optimizer='adam', loss='mean_squared_error')
  return model

model1 = get_model()
model2 = get_model()
#model2 = get_model()

X_audio = pickle.load( open( "audio.txt", "rb" ) )
X_word = pickle.load( open( "word_original.txt", "rb" ) )
Y1 = pickle.load( open( "Y1_original.txt", "rb" ) )
Y2 = pickle.load( open( "Y2_original.txt", "rb" ) )

X_audio = np.reshape(X_audio, (13860, 10000, 1))
# Y1 = np.reshape(Y1, (13860, 30, 1))
# Y2 = np.reshape(Y2, (13860, 30, 1))

X_audio, X_word, Y1, Y2 = shuffle(X_audio, X_word, Y1, Y2, random_state=0)

#model = load_model('new_model_scaled.h5')

dic_audio = {}
dic_word = {}
dic_y1 = {}
dic_y2 = {}

Y1 = Y1 + 500
Y2 = Y2 - 500
Y1 = Y1/1000
Y2 = Y2/1000
for x_audio, x_word, y1, y2 in zip(X_audio, X_word, Y1, Y2):
	le = len(y1)
	if dic_audio.get(le) is None:
		dic_audio[le] = [x_audio]
		dic_word[le] = [x_word]
		dic_y1[le] = [y1]
		dic_y2[le] = [y2]
	else:
		dic_audio[le].append(x_audio)
		dic_word[le].append(x_word)
		dic_y1[le].append(y1)
		dic_y2[le].append(y2)

for i in range(0, 1):
	for j in dic_audio.keys():
		x_aud = np.array(dic_audio[j])
		x_word = np.array(dic_word[j])
		print(x_word.shape)
		y2 = np.array(dic_y2[j])
		y2 = np.reshape(y2, (y2.shape[0], y2.shape[1], 1))
		y1 = np.array(dic_y1[j])
		y1 = np.reshape(y1, (y2.shape[0], y2.shape[1], 1))
		#y = [y1, y2]
		#y = np.array(y)
		#y = np.reshape(y, (y.shape[1], y.shape[2], 2))


		# y = model.predict([x_aud, x_word])
		#print(y1)
		#print(y)
		history1 = model1.fit([x_aud, x_word], y1, batch_size = 50, epochs = 10)
		model1.save('6_length_start.h5')
		history2 = model2.fit([x_aud, x_word], y2, batch_size = 50, epochs = 10)
		
		model2.save('6_length_end.h5')
#		with open("history1.txt","wb") as file:
#			pickle.dump(history1,file)
#		with open("history2.txt","wb") as file:
#			pickle.dump(history2,file)
		
#		model2.fit([x_aud, x_word], y2, batch_size = 50, epochs = 2)
		break

#model1.save('new_model_scaled_1000_start.h5')
#model2.save('new_model_scaled_1000_end.h5')


model1.save('6_length_start_tweak2.h5')
model2.save('6_length_end_tweak2.h5')





