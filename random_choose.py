from random import randint
import utils
import os
import msgpack

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
(train_x, train_y), (test_x, test_y) = utils.read_data(FILE_PATH)

forumKeys = msgpack.unpack(open('./input/forumKeys.msgpack', 'rb'))

for i in range(0, 5):
	rand = randint(0, len(forumKeys)-1)
	predicted = forumKeys[rand]

	if predicted[2] == train_y[i].tolist():
		print('Match:' + forumKeys[rand][1].decode('ascii') + '\n')