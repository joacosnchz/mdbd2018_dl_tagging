import msgpack
import matplotlib.pyplot as plt
import numpy as np
import statistics

def histogram(file, title=None):
	wC = msgpack.unpack(open(file, 'rb'))
	wD = msgpack.unpack(open('./input/wordDict_merged.msgpack', 'rb'))

	toDelete = []
	for i in range(0, len(wC)):
		if (wC[i] < 3 or len(wD[i]) > 28 or '\\' in str(wD[i])) or wC[i] > 4000:
			toDelete.append(i)

	for i in toDelete[::-1]:
		wC.pop(i)

	print(len(wC))

	mean = statistics.mean(wC)
	sd = statistics.stdev(wC)
	print(mean, sd)

	plt.hist(wC)
	if title:
		plt.title(title)
	plt.ylabel('Frequency')
	plt.xlabel('Word Ocurrences')
	plt.show()

def xy_lineplot(file, title=None):
	x = []
	y = []
	with open(file) as f:
		lines = f.readlines()

		for i in range(1, len(lines)):
			splittedline = lines[i].split('\t')

			x.append(splittedline[0])
			y.append(float(splittedline[1]))

	plt.plot(x, y)
	if title:
		plt.title(title)
	plt.xlabel('Epochs')
	plt.ylabel('Avg. Accuracy')
	#plt.xticks([0, 19, 39, 59, 79, 99])
	plt.show()

#xy_lineplot('./input/bs_ma.txt', 'Batch Size vs Model Accuracy')
#xy_lineplot('./input/training.txt', 'Epochs Quantity vs Model Accuracy')
#xy_lineplot('./input/bs_time.txt', 'Batch Size vs Time (min)')
#xy_lineplot('./input/100trainings.txt', 'Epochs vs Avg. Accuracy')
histogram('./input/wordCount_merged.msgpack', 'Histogram of Word Ocurrences')
