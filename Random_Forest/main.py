import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import streamlit as st                #for building web-app
from sklearn.datasets import load_iris
 
#DATA

@st.cache_data                           #caches the result so dataset doesn't reload every time the app returns
def load_data():                         #loads the dataset (150 sample, 3species)
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target         #Adds class labels (divide the only column sepal/petal length/width data into rows and column) 
    return df, iris.target_names        #convert prediction result (like[0]) to label name (like setosa)

df, target_names = load_data()


#MODEL
model = RandomForestClassifier()  

#TRAIN
model.fit(df.iloc[:, :-1], df['species'])  # df.iloc[:, :-1] selects all the rows and column features(sepal/petal length/width) 
                                           # of species [setosa', 'versicolor', 'virginica']



#INPUT : CREATES SLIDERS TO TAKE INPUTS(SEPAL LENGTH, WIDTH, PETAL LENGTH, WIDTH)

st.sidebar.title("Input Features")  # creates slider input widget in Streamlit app

sepal_length = st.sidebar.slider("Sepal length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))


input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

#PREDICTION

prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]  # converts predicted no label into actual flower name


#RESULT
st.write("Prediction")
st.write(f"The predicted species is : {predicted_species}")



#  Sepal length: ~5.1
#  Sepal width: ~3.5          for Setosa
#  Petal length: ~1.4
#  Petal width: ~0.2


#  Sepal length: ~6.0
#  Sepal width: ~2.8          for Versicolor
#  Petal length: ~4.7 
#  Petal width: ~1.4


#  Sepal length: ~6.5+
#  Sepal width: ~3.0          for Virginca
#  Petal length: ~5.5+
#  Petal width: ~2.0+

#cd / home/flash/Desktop/ML/Random_Forest/
#Streamlit run main.py