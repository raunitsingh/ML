import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier   #fimport neighbor module
from sklearn.datasets import load_iris              #imports iris flower dataset
from sklearn.model_selection import train_test_split   #split module into Training and Testing set
from sklearn.metrics import classification_report, accuracy_score  #gives prediction and precision


#Dataset
data = load_iris()
x = data.data
y = data.target
target_names = data.target_names


# Split into training and testing (80/20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #fixes the randomness to avoid re-shuffling again and again


#KNN Model
model = KNeighborsClassifier(n_neighbors=3)


#Train
model.fit(x_train, y_train)


#Predict
y_pred = model.predict(x_test)

#Result
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))

