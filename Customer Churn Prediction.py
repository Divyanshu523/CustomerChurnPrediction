import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense


#Reading the file from the location
df=pd.read_csv("C:\Users\Divyanshu.LAPTOP-FO77DCME\Downloads\Churn_Modelling.csv")
#Some of the basic functions in panadas
df.head()
df.drop(columns=['RowNumber','CustomerId','Surname'],inplace=True)
df.head()

# selecting the data or field on which the model will be trained
df['Geography'].value_counts()
df['Gender'].value_counts()

df = pd.get_dummies(df,columns=['Geography','Gender'],drop_first=True)

df.head()

X = df.drop(columns=['Exited'])
y = df['Exited'].values

# spliting the data into training and testing data 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


scaler = StandardScaler()

X_train_trf = scaler.fit_transform(X_train)
X_test_trf = scaler.transform(X_test)


#creating the neural network to make the predictons
model = Sequential()

model.add(Dense(11,activation='sigmoid',input_dim=11))
model.add(Dense(11,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
#created the layers for the ANN and now defing the epoch properties 
history = model.fit(X_train,y_train,batch_size=50,epochs=100,verbose=1,validation_split=0.2)
y_pred = model.predict(X_test)
y_pred
y_pred = y_pred.argmax(axis=-1)

#giving the accuracy of the trained model
accuracy_score(y_test,y_pred)


#plotting different values on graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
