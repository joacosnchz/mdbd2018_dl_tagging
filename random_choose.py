from random import randint
import utils
import os
import msgpack

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
(train_x, train_y), (test_x, test_y) = utils.read_data(FILE_PATH)

forumKeys = msgpack.unpack(open('./input/forumKeys.msgpack', 'rb'))


with open('./output/random_accuracy.txt', 'w') as f:
	for r in range(0, 100):
		correct = 0
		total_comparations = len(train_y)
		for i in range(0, total_comparations):
			rand = randint(0, len(forumKeys)-1)
			predicted = forumKeys[rand]

			if predicted[2] == train_y[i].tolist():
				correct += 1

		accuracy = correct/total_comparations
		print('Random accuracy: ' + str(accuracy))
		f.write(str(r+1) + '\t' + str(accuracy) + '\n')