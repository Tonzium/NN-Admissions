import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

df = pd.read_csv("Admissions_data.csv")

print(df.columns)
print(df.describe())
print(df.dtypes)

# remove only last spaces
df.columns = df.columns.str.rstrip()

features = df.drop("Chance of Admit", axis=1)
labels = df["Chance of Admit"]

# scaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
df_scaled = pd.DataFrame(features_scaled)

features_train, features_test, labels_train, labels_test = train_test_split(df_scaled, labels, test_size=0.2, random_state=9)

my_input = InputLayer(input_shape = (features_train.shape[1], ))

# model
model= Sequential()
model.add(my_input)
model.add(Dense(64,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
print(model.summary())

# Compile the model
opt = Adam(learning_rate = 0.011)
model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)

# EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=4)

# fit model
history = model.fit(features_train, labels_train, epochs=40, validation_split=0.2, callbacks=[early_stopping], verbose = 1)

# evaluate model
final_mse, final_mae = model.evaluate(features_test, labels_test, verbose = 0)
print("Result mse:", final_mse)
print("Result mae:", final_mae)

y_pred = model.predict(features_test)

score = r2_score(labels_test, y_pred)
print("R-squared score:", score)

# plot model evaluation

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
 
# Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')
 
fig.tight_layout()
fig.savefig('my_plots.png')