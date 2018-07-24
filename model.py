# *********************************************
# Diabetic Retinopathy Prediction using Keras

# Yug Khanna - 24 July 2018
# *********************************************



import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense
from keras.models import model_from_json

seed = 3

np.random.seed(seed) # to get constant results

dataset = np.loadtxt("messidor_features.csv", delimiter=",") #loading dataset, you can use your own dataset

train, test = train_test_split(dataset, test_size=0.2) #splitting data for testing and training

X = train[:,0:19] #creating input variable
Y = train[:,19] #creating output variable

scaler = MinMaxScaler()
scaler.fit(dataset)
dataset = scaler.transform(dataset)

model = Sequential()
model.add(Dense(19, input_dim=19,activation='relu', init="uniform"))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# relu->tanh->relu->sigmoid(0 or 1)

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=300, batch_size=20) #you can alter this according to your needs. Hit and Trial works :)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)) #Getting 85% Train Accuracy

model_json = model.to_json()
with open('model.json','w') as json_file:
    json_file.write(model_json) #Saving the model

model.save_weights('model.h5') #saving weights

print("Model Saved!")

X_Test = test[:,0:19]
Y_Test = test[:,19]

predictions = model.predict(X_Test)

rounded = [round(x[0]) for x in predictions]
print(rounded)

scores = model.evaluate(X_Test, Y_Test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))  #Test Accuracy = 76%
