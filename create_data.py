import numpy as np
import os
import librosa
import soundfile as sf
from pathlib import Path
import pickle
import random

def getGloVector(glove_vectors_file = "glove.6B.50d.txt"):
	glove_wordmap = {}
	with open(glove_vectors_file, "r", encoding="utf8") as glove:
	    for line in glove:
	        name, vector = tuple(line.split(" ", 1))
	        glove_wordmap[name] = np.fromstring(vector, sep=" ")
	return glove_wordmap



def fill_unk(unk):
	global glove_wordmap
	glove_wordmap[unk] = RS.multivariate_normal(m,np.diag(v))
	return glove_wordmap[unk]


glove_wordmap = getGloVector()
wvecs = []
for item in glove_wordmap.items():
    wvecs.append(item[1])

s = np.vstack(wvecs)

# Gather the distribution hyperparameters
v = np.var(s,0) 
m = np.mean(s,0) 
RS = np.random.RandomState()

def getWordBoundariesAndWords(file_name):
  f = open(file_name, "r")
  content = f.readlines()
  words = list()
  start = list()
  end = list()
  for line in content:
    tokens = line.split()
    start.append(tokens[0])
    end.append(tokens[1])
    words.append(tokens[2].lower())
    # if vocab_dict.get(tokens[2].lower()) is None:
    #   vocab_dict[tokens[2].lower()] = True

  return words, start, end

def getWordVector(word):
	# if word_map.get(word) is None:
	if glove_wordmap.get(word) is None:
		wrd_map = fill_unk(word)
		# word_map[word] = wrd_map
	else:
		wrd_map = glove_wordmap[word]
	return wrd_map


def read_wav(file_name):
  data, samplerate = sf.read(file_name)
  return data

dir_map = {}


def createDirMap():
	count = 0
	X_audio = list()
	X_word = list()
	Y1 = list()
	Y2 = list()
	dir_list = [x[0] for x in os.walk("./test")]
	print(len(dir_list))

	for dire in dir_list:
		folder_lis = [x[0] for x in os.walk(dire+"/")]
		# print(folder_lis)
		for folder in folder_lis:
			file_dict = {}
			# print(dire, folder)
			for filename in os.listdir(folder+"/"):
				if filename.endswith(".wav"):
					key = filename.split(".")
					file_dict[key[0]] = True

			for file_name in file_dict.keys():
				file_dir = folder+"/"+file_name
				wav_file = file_dir+".wav"
				wrd_file = file_dir+".wrd"
				my_file = Path(wrd_file)
				if not my_file.is_file():
					continue
				data = read_wav(wav_file)
				data = np.array([ data[i] for i in sorted(random.sample(range(len(data)), 10000)) ])
				words, start, end = getWordBoundariesAndWords(wrd_file)
				word_lis = list()
				for word in words:
					wrd_vec = getWordVector(word)
					word_lis.append(np.array(wrd_vec))
				#while len(word_lis) < 30:
				#	word_lis.append(np.zeros(50))
				#	start.append(0)
				#	end.append(0)
				  
				X_audio.append(data)
				  
				X_word.append(np.array(word_lis, dtype=np.float32))
				Y1.append(np.array(start, dtype=np.float32))
				Y2.append(np.array(end, dtype=np.float32))
				# print (count)
				# count += 1
	X_audio = np.array(X_audio)
	X_word = np.array(X_word)
	Y1 = np.array(Y1)
	Y2 = np.array(Y2)
	print(X_audio.shape, X_word.shape, Y1.shape, Y2.shape)
	return X_audio, X_word, Y1, Y2
			  
			


X_audio, X_word, Y1, Y2 = createDirMap()

with open("audio_test.txt","wb") as file:
	pickle.dump(X_audio,file)
with open("word_original_test.txt","wb") as file:
	pickle.dump(X_word,file)
  
with open("Y1_original_test.txt","wb") as file:
	pickle.dump(Y1,file)
  
with open("Y2_original_test.txt","wb") as file:
	pickle.dump(Y2,file)



