# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#1. Import the necessary packages and dataset
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, TensorBoard
import datetime, os
import pandas as pd

path = r"C:\Users\muhdi\Documents\Deep Learning\Heart-Disease-Project\data\heart.csv" #paste the path of the downloaded csv
df = pd.read_csv(path)

#%%
#2. check DF if it needs any cleaning
df.head
print(df.isna().sum())
#data is clean, hence can go on to next step

#%%
#3. Split data into features and labels
features = df.copy()
labels = features.pop('target')

#convert to np array to be able to run training
features = np.array(features)
labels = np.array(labels)

#%%
#4. Split the features and labels into train-validation-test sets
SEED = 12345
x_train, x_iter, y_train, y_iter = train_test_split(features,labels,test_size=0.4,random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_iter,y_iter,test_size=0.5,random_state=SEED)

#Perform data normalization
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#%%
#5. Build a FFNN for classification
no_input = x_train.shape[-1]
no_output = len(np.unique(y_test))

model = keras.Sequential()
model.add(layers.InputLayer(input_shape=no_input))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(32,activation='relu'))
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(no_output,activation="softmax"))

#view your model
model.summary()

#to view a representation of your model structure
tf.keras.utils.plot_model(model, show_shapes=True)

#%%
#6. Compile your model and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#7. Define callback functions 
base_log_path = r"C:\Users\muhdi\Documents\Deep Learning\Heart-Disease-Project\logs"
log_path= os.path.join(base_log_path, datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '__Heart_Disease_Dataset')
#set patience of 15 epochs before stopping once there is no significant improvement of model training
es = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
#use TensorBoard to view your training graph
tb = TensorBoard(log_dir=log_path)
BATCH_SIZE = 32

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=BATCH_SIZE, epochs=100,callbacks=[es,tb])

#%%
#to view graph of train against validation, run in the code below without the # key in prompt
#tensorboard --logdir "C:\Users\muhdi\Documents\Deep Learning\Heart-Disease-Project\logs"

#%%
#Evaluate with test data for wild testing
test_result = model.evaluate(x_test,y_test,batch_size=BATCH_SIZE)
print(f"Test loss = {test_result[0]}")
print(f"Test accuracy = {test_result[1]}")
