import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import msgpack
import numpy as np
import random

import tensorflow as tf
from tensorflow.python.client import device_lib

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
	
	'''
	print('Reading data...')
	folders = [f[2:] for f in[i[0] for i in os.walk(file_path  + '/input' )][1:]]
	posts = []
	indexes = []

	for f in folders:
		forumname = f.split('.')[0].title()
		forumnamesplitted = forumname.split('/')
		indexes.append(forumnamesplitted[len(forumnamesplitted)-1]) #Take forum's name
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

	return ((x_train,y_train),(x_test,y_test),indexes)


def plot_confusion_matrix(df, save=False):
	''' Recieves the dataframe and plots it.
	'''
	plt.figure(figsize=(8,6))

	#generate heatmap axis and styles
	ax = sns.heatmap(df, linewidths=.5)
	plt.subplots_adjust(left=.20, bottom=.21,  top=.94)
	ax.set_title("Confusion Matrix", fontsize='large')
	ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode='anchor')

	if save:
		plt.savefig('./output/confusion_matrix.png')
	plt.show()


def use_neural_network(input_data):
    prediction = model.predict(x)
    lemmatizer = WordNetLemmatizer()

    with open('./input/WordDict.msgpack', 'rb') as f:
        lexicon_pre = msgpack.unpack(f)
        lexicon = []
        for word in lexicon_pre:
            lexicon.append(word.decode())

        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './prueba/model_trains' + '_epochs_' + HM_EPOCHS + '_batch_' + BATCH_SIZE + '/model.ckpt')

        features = tags_worddict.get_multihot(input_data)
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
        
        print('Result', result[0])
