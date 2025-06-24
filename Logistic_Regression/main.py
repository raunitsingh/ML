import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression

#Data:  0 = Fail &  1 = Pass
x = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

#model
model = LogisticRegression()

#train
model.fit(x,y)

#Prediction for 4.5 hrs
prob = model.predict_proba([[4.5]])[0]
print(f"Probability of Passing with 4.5 hrs: {prob[1]*100:.2f}%")


#Prediction of class 0 or 1
print("Prediction:",model.predict([[4.5]])[0])


#plot sigmoid curve

x_val = np.linspace(0, 7, 100).reshape(-1,1)  #Generates test data points for each hours
y_prob = model.predict_proba(x_val)[:, 1]     #Gets probabilities for each hour


plt.scatter(x, y, color='red', label= 'actual data')
plt.plot(x_val, y_prob, color='blue', label='Pass Probability')
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression - Sigmoid Curve")
plt.grid(True)
plt.legend()
plt.show()


