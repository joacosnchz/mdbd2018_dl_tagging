import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import msgpack
import numpy as np
import random

from tensorflow.python.client import device_lib

import matplotlib.pyplot as plt

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def read_data(file_path):
	'''
	Simple BoW model to classify questions
	Determines whether each word appears in a post, but it does not understand grammar or sequence
	A comment is represented by a dense multi-hot vector of the vocabulary size
	
	Vocab = [‘listview’, ‘strftime’, ‘studio’, ‘isnan’, ‘script’]
	“How to make a ListView in Android Studio” => [1 0 1 0 0]
	Limit the vocab size for the model to the top 1000 most commonly used words (Esto hay que cambiarlo)
	'''
	print('Reading data...')
	folders = [f[2:] for f in[i[0] for i in os.walk(file_path  + '/input' )][1:]]

	posts = []
	for f in folders:
		with open('/U' + f + '/OneHot.msgpack', 'rb') as input:
			posts +=  (msgpack.unpack(input))

	posts = np.array(posts)
	random.shuffle(posts)

	#90% of data for training and 10% for testing
	train_size = int(len(posts) * .9)

	print('Splitting train data...')
	x_train = np.array(posts[:train_size][:,0].tolist())
	y_train = np.array(posts[:train_size][:,1].tolist())

	print('Splitting test data...')
	x_test = np.array(posts[train_size:][:,0].tolist())
	y_test = np.array(posts[train_size:][:,1].tolist())

	del posts

	return ((x_train,y_train),(x_test,y_test))



def plot_history(histories, key='categorical_crossentropy'):
	'''Plot history graph for loss and train per epoch.
	   Analyse overfitting.
	'''

	plt.figure(figsize=(8,5))
    
	for name, history in histories:
		val = plt.plot(history.epoch, history.history['val_'+key],
					'--', label=name.title()+' Val')
		plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
					label=name.title()+' Train')

	plt.xlabel('Epochs')
	plt.ylabel(key.replace('_',' ').title())
	plt.legend()

	plt.xlim([0,max(history.epoch)])
	plt.show()


#print(device_lib.list_local_devices()[:,"name"])