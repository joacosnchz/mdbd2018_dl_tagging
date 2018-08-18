import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 5

def create_lexicon(files):
	lexicon = []
	for fi in files:
		with open(fi, 'r') as f:
			contents = f.readlines()
			for l in contents[:hm_lines]:
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]

	w_counts = Counter(lexicon)
	l2 = []
	for w in w_counts:
		#if 1000 > w_counts[w] > 50:
			l2.append(w)

	with open('./input/lexicon.pickle', 'wb') as f:
		pickle.dump(l2, f)
	print(len(l2))
	return l2

def sample_handling(sample, lexicon, classification):
	featureset = []
	with open(sample, 'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))

			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1

			features = list(features)
			featureset.append([features, classification])

	return featureset

def create_feature_sets_and_labels(test_size=0.1):
	files = [
		'./input/apache.txt',
		'./input/php.txt',
		'./input/pip.txt',
		'./input/python.txt'
	]
	lexicon = create_lexicon(files)
	features = []
	features += sample_handling(files[0], lexicon, [1, 0, 0, 0])
	features += sample_handling(files[1], lexicon, [0, 1, 0, 0])
	features += sample_handling(files[2], lexicon, [0, 0, 1, 0])
	features += sample_handling(files[3], lexicon, [0, 0, 0, 1])
	random.shuffle(features)

	features = np.array(features)

	testing_size = int(test_size*len(features))

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])

	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x, train_y, test_x, test_y



