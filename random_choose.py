from random import randint
import utils
import os

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
(train_x, train_y), (test_x, test_y) = utils.read_data(FILE_PATH)

print(train_y)

rand = randint(1, 15)

classes = {
	1: '3dPrinting', 
	2: 'chess', 
	3: 'esperanto', 
	4: 'fitness', 
	5: 'freelancing', 
	6: 'health', 
	7: 'law', 
	8: 'pets', 
	9: 'robotics', 
	10: 'rpg', 
	11: 'sports', 
	12: 'startups', 
	13: 'vi', 
	14: 'windowsPhone',
	15: 'woodWorking'
}

print(classes[rand])