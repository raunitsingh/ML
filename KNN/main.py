import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier


#Data Features: [weight in gm, size in cm(diameter)]
x = np.array([
    [150,7],       #orange
    [170, 7.5], 
    [180, 8],
    [130, 6.5],    #apple
    [120, 6],
    [110, 5.5]
])
y = np.array([0, 0, 0, 1, 1, 1])   # 0 = Orange, 1 = Apple


#model
model = KNeighborsClassifier(n_neighbors=3)  #scanning nearby 3 neighbors


#Train
model.fit(x,y)

#prediction
fruit = np.array([[100, 7]])
result = model.predict(fruit)

print("Predicted label:", "Apple" if result[0] == 1 else "orange")