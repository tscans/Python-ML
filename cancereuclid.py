import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

#euclidian_distance = sqrt((plot1[0]-plot2[0])**2+(plot1[1]-plot2[1])**2)


# [[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]

# plt.scatter(new_features[0], new_features[1])
# plt.show()

def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting groups')

	distances = []
	#data is basically the dataset and group is each class
	for group in data:
		for features in data[group]:
			#euclidian_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
			euclidian_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidian_distance, group])
	votes = [i[1] for i in sorted(distances)[:k]]
	#print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = float(Counter(votes).most_common(1)[0][1]) / float(k)
	#print(vote_result, confidence)
	return vote_result, confidence
accuracies = []

for i in range(25):

	df = pd.read_csv('cancer.txt')
	#adjusting unknown data
	df.replace('?', -99999, inplace=True)
	#dropping id
	df.drop(['id'], 1, inplace=True)
	#to remove quotes
	full_data = df.astype(float).values.tolist()

	#to shuffle the data
	random.shuffle(full_data)

	test_size = 0.2
	train_set = {2:[], 4:[]}
	test_set = {2:[], 4:[]}
	#getting data
	train_data = full_data[:-int(test_size*len(full_data))]
	test_data = full_data[int(test_size*len(full_data)):]

	for i in train_data:
		train_set[i[-1]].append(i[:-1])
	for i in test_data:
		test_set[i[-1]].append(i[:-1])

	correct = 0
	total = 0

	for group in test_set:
		for data in test_set[group]:
			vote, confidence = k_nearest_neighbors(train_set, data, k=5)
			if group == vote:
				correct+=1
			total += 1

	accurate = float(correct)/float(total)
	print('Accuracy: ', accurate)

	print(correct)
	print(total)
	accuracies.append(accurate)

print(sum(accuracies)/len(accuracies))
















