import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

#import the file
df = pd.read_csv('cancer.txt')
#to remove unknown data
df.replace('?', -99999, inplace=True)
#to get rid of id column to avoid tmpering with data, if we didnt do this the accuracy would be like 50
df.drop(['id'], 1, inplace=True)

#x for attributes, takes all columns but last
X = np.array(df.drop(['class'],1))
#y for class, takes just the last column
y = np.array(df['class'])

#doing cross_validation, splitting the data and then using 20% of the data to train
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)

#classifier
clf = neighbors.KNeighborsClassifier()
#fitting the training data
clf.fit(X_train, y_train)

#we are calling it accuracy because we measure conf as something else
accuracy = clf.score(X_test, y_test)
print(accuracy)

#example to predict, we are making an example data point
#made another one
example_measures = np.array([[4,2,1,1,1,2,3,2,1], [10,2,2,2,2,3,3,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1)

#this is the machine learning algorithm in action, it takes my data points and predicts whether the cancer is benign or malignant with a 2 or 4
prediction = clf.predict(example_measures)
print(prediction)









