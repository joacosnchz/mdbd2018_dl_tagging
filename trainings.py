import os

hm_trainings = 9
batch_size = 10

for i in range(0, hm_trainings):
	# batch_size hm_epochs is_training n_classes
	os.system('python tags_net.py ' + str(batch_size) + ' 100 1 15 ' + str(i))
	batch_size += 10
	# SET /A N_EPOCHS=!N_EPOCHS!+2