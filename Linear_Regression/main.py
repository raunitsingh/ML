import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


#Data: Hours studied vs Score
x = np.array([[1], [2], [3], [4], [5] ])
y = np.array([50, 55, 65, 70, 75])

#create model
model = LinearRegression()

#Train the model
model.fit(x,y)

#Predict
y_pred = model.predict(x)


#plot the results
plt.scatter(x,y, color='blue', label='actual data')
plt.plot(x,y_pred, color='red', label='prediction line')
plt.xlabel('Hours studied')
plt.ylabel('Test score')
plt.title('Linear Regression')
plt.legend()
plt.show()



print("slope(m):", model.coef_[0])
print("Intercept(b):", model.intercept_)
print("prediction for 6 hours study:", model.predict([[6]])[0])
