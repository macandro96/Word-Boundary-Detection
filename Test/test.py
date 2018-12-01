# Code credits https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

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



def metrics(y1_true, y2_true, y1_pred, y2_pred):
	#print(y1_true.shape, y1_pred.shape)
	number_preds = list()
	std = 0
	mean = 0
	ex2 = 0
	for start_true,end_true in zip(y1_true, y2_true):
		count = 0
		temp_start = 0
		temp_end = 0

		for start_pred, end_pred in zip(y1_pred, y2_pred):
			if start_pred >= start_true and end_pred <= end_true:
				count = count + 1
				temp_start = abs(start_pred - start_true)
				temp_end = abs(end_pred - end_true)
		if count == 1:
			ex2 = ex2 + (temp_start)**2 + (temp_end)**2
			mean = mean + temp_start + temp_end
		number_preds.append(count)
	correct_detection = 0
	false_detection = 0
	missed_detection = 0
	for num in number_preds:
		if num == 1:
			
			correct_detection += 1
		elif num > 1:
			false_detection += 1
		else:
			missed_detection += 1
	tot = len(number_preds)
	var = ex2/tot - (mean/tot)**2
	return (correct_detection, false_detection, missed_detection, var)

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

#model = get_model()
#model2 = get_model()

X_audio = pickle.load( open( "audio_test.txt", "rb" ) )
X_word = pickle.load( open( "word_original_test.txt", "rb" ) )
Y1 = pickle.load( open( "Y1_original_test.txt", "rb" ) )
Y2 = pickle.load( open( "Y2_original_test.txt", "rb" ) )

X_audio = np.reshape(X_audio, (5040, 10000, 1))
# Y1 = np.reshape(Y1, (13860, 30, 1))
# Y2 = np.reshape(Y2, (13860, 30, 1))

X_audio, X_word, Y1, Y2 = shuffle(X_audio, X_word, Y1, Y2, random_state=0)

model1 = load_model('6_length_start_tweak.h5')
model2 = load_model('6_length_end_tweak.h5')

dic_audio = {}
dic_word = {}
dic_y1 = {}
dic_y2 = {}

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
tot = 1
tot_cd = 0
tot_fd = 0
tot_md = 0
k = True
std_tot = 0
std_mean = 0
for i in range(0, 1):
	for j in dic_audio.keys():

		x_aud = np.array(dic_audio[j][0:200])
		x_word = np.array(dic_word[j][0:200])
		if x_word.shape[1] is not 17:
			continue
		print(x_word.shape)
		y1 = np.array(dic_y1[j][0:200])
		y1 = np.reshape(y1, (y1.shape[0], y1.shape[1], 1))
		y2 = np.array(dic_y2[j][0:200])
		y2 = np.reshape(y2, (y2.shape[0], y2.shape[1], 1))
		y1_pred = model1.predict([x_aud, x_word])
		y2_pred = model2.predict([x_aud, x_word])
		#y_input = np.zeros((y1.shape[0],y1.shape[1],2))
		#y_input[:,:,0] = np.reshape(y1, (y1.shape[0] , y1.shape[1]))
		#y_input[:,:,1] = np.reshape(y2, (y2.shape[0] , y2.shape[1]))
		#y_input_pred = np.zeros((y1.shape[0],y1.shape[1],2))
		#y_input_pred[:,:,0] = np.reshape(y1_pred, (y1_pred.shape[0] , y1_pred.shape[1]))
		#y_input_pred[:,:,1] = np.reshape(y2_pred, (y2_pred.shape[0] , y2_pred.shape[1]))
		#print("Old shape 1 ",y1.shape)
		#print("Old shape 2 ",y2.shape)
		#print("New shape ", (np.array([y1, y2])).shape)
		#print("Done ", y_input.shape)
		#cd, fd, md = correct_false_miss_rate(y_input, y_input_pred, x_word)
		#print("counts",cd, fd, md)
		#break
		for k in range(0, x_aud.shape[0]):
			y1_true  =y1[k]
			y2_true = y2[k]
#			print(y1_pred.shape)
			y1_pred_ = y1_pred[k]
			y2_pred_ = y2_pred[k]

			cd,fd,md,std = metrics(y1_true, y2_true, y1_pred_, y2_pred_)
			prob = cd/(cd + fd + md)
			std_tot += std*std/x_aud.shape[0]
			std_mean += std/x_aud.shape[0]
#			cd, fd, md = correct_false_miss_detection(np.array([y1_true, y2_true]), np.array([y1_pred_, y2_pred_]), )
			tot = tot + (cd + fd + md)
			tot_cd += cd
			tot_fd += fd
			tot_md += md
#			print(k)

		break			
		

cdr = tot_cd/tot
fdr = tot_fd/tot
mdr = tot_md/tot
std = std_tot - std_mean**2
print(cdr, fdr, mdr, std, std_tot, std_mean)

		#model.fit([x_aud, x_word], y1, batch_size = 50, epochs = 2)
#		model2.fit([x_aud, x_word], y2, batch_size = 50, epochs = 2)


#model1.save('6_length_start.h5')
#model2.save('6_length_end.h5')


#model.save('new_my_model_combined.h5')

