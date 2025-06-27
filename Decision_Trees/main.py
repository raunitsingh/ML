import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Data: [Income, Age, LoanAmount]
x = np.array([
    [40, 25, 10],
    [50, 30, 20],
    [60, 35, 25],
    [35, 45, 15],
    [20, 23, 5],
    [90, 40, 30],
    [70, 50, 10],
    [30, 28, 8]

])

#Loan Approval: 0 = no & 1 = yes
y = np.array([1, 1, 1, 0, 0, 1, 1, 0])

#Train-Test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


#Model
model = DecisionTreeClassifier(max_depth=3)

#Train
model.fit(x_train, y_train)

#Predict
y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


#Other values
new_customer = [[55, 32, 20]]
decision = model.predict(new_customer)[0]
print("Loan Approval" if decision ==1 else "Loan Denied")


