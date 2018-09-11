import os

hm_trainings = 1
batch_size = 10

for i in range(0, hm_trainings):
	# batch_size hm_epochs is_training n_classes
	os.system('python3 tags_net.py ' + str(batch_size) + ' 2 1 2 ' + str(i))
	batch_size += 10