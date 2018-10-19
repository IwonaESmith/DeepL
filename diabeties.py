#  Keras application for diabetes problem


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random


from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv("pima-indians-diabetes.csv", header=None, delimiter=",", dtype=float)

df1 = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (x) and output (y) variables
X = df.iloc[:,0:8].values
y = df.iloc[:,8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

dims = x.shape[1]

# create model
model = Sequential()
# Adding the input layer and the first hidden layer
model.add(Dense(12, input_dim = dims, activation = 'relu'))
# Adding the input layer and the second hidden layer
model.add(Dense(8, input_dim=8, activation='relu'))
# Adding the output layer
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# Fit the model
model.fit(X_train, y_train, epochs=150, batch_size=10)

# Evaluate the model
scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Calculate predictions
y_pred = model.predict(X_test)
rounded = [round(x[0]) for x in y_pred]
print(rounded)

y_test.astype(float)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred.round())
print(cm)

