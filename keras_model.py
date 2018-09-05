
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np

import tensorflow as tf
from tensorflow import keras


import utils

# Model hyperparameters
batch_size = 128
LABELS_SIZE = 2
VOCAB_SIZE = 20736		#Number of words in the dictionary  (Limit the vocab size for the model to the top 1000 most commonly used words)

#Data parameters
FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def build_model():
	'''
	Stack of layers:
		The input layer takes the vocab_size arrays for each post.
		Implement a layer dropout to prevent overfitting
		Last layer (output layer) take the “bag of words” for each post and output a 20-element array indicating the probability that the question belongs to each of our 20 tags
		Implement Softmax: normalize the evidence for each possible label into a probability (from 0 to 1), and these 2 values for a given post will sum up to 1.
	'''
	model = keras.Sequential([keras.layers.Dense(batch_size, activation=tf.nn.relu, input_shape=(VOCAB_SIZE,)),
							#keras.layers.Dropout(0.5),
							keras.layers.Dense(LABELS_SIZE, activation=tf.nn.softmax),
							#keras.layers.Dropout(0.5),
							])
	return model


def training(model, x_train, y_train):
	'''
	Call the compile method with the loss function, the type of optimizer, and the metrics the model should evaluate during training and testing.
	Since each of our comments can only belong to one post, use cross-entropy.
	Adamoptimizer as the Optimizer.
	For metrics, evaluate the accuracy for the percentage of comments it assigned the correct label to
	'''
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #metrics=['MAE','MSE']

	history = model.fit(x_train, y_train, 
					batch_size=batch_size, 
					epochs=13, 
					verbose=1, 
					validation_split=0.1)

	return history



def evaluate_scoring(model, x_test, y_test):
	'''Evaluates the accuracy of the trained model on the 20% of the data'''
	score = model.evaluate(x_test, y_test, 
                       batch_size=batch_size, verbose=1)
	
	print('\nTest score:', score[0])
	print('Test accuracy:', score[1])



if __name__ == '__main__':
	utils.safe_mkdir('checkpoints')

	#Fetch the data
	(x_train,y_train),(x_test,y_test) = utils.read_data(FILE_PATH)

	#Build the model
	model = build_model() 

	#Train the model
	history = training(model, x_train, y_train) 

	evaluate_scoring(model, x_test, y_test)

	#Compare the loss function for both and check overfitting
	utils.plot_history([('Crossentropy', history)], "acc")
