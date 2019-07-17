from __future__ import absolute_import, division, print_function
import pathlib
import numpy as no
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset_path = pd.read_csv('Churn_Modelling.csv')
dataset_path.drop(dataset_path.columns[[0,1,2]], axis=1, inplace = True)
# print(dataset_path)
data_num = dataset_path._get_numeric_data()
data_cat = dataset_path.select_dtypes(exclude=["number"])

def d(x):
	if x == 'France' :
		return 1
	else:
		return 0
dataset_path['France'] = dataset_path['Geography'].map(d)

def d(x):
	if x == 'Spain' :
		return 1
	else:
		return 0
dataset_path['Spain'] = dataset_path['Geography'].map(d)

def d(x):
	if x == 'Germany' :
		return 1
	else:
		return 0
dataset_path['Germany'] = dataset_path['Geography'].map(d)
del dataset_path['Geography']

def d(x):
	if x == 'Male' :
		return 1
	else:
		return 0
dataset_path['Male'] = dataset_path['Gender'].map(d)
del dataset_path['Gender']

train_dataset = dataset_path.sample(frac=0.8,random_state=0)
test_dataset = dataset_path.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("CreditScore")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('CreditScore')
test_labels = test_dataset.pop('CreditScore')

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
	model = keras.Sequential([
		layers.Dense(12, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
		layers.Dense(12, activation=tf.nn.relu),
		layers.Dense(1)])
	optimizer = tf.keras.optimizers.RMSprop(0.001)
	model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['mean_absolute_error', 'mean_squared_error'])
	return model
model = build_model()
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)

class PrintDot(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		if epoch % 100 == 0:
			print('')
			print('.', end='')

EPOCHS = 100

history = model.fit(normed_train_data, train_labels,epochs=EPOCHS, validation_split = 0.2, verbose=0,callbacks=[PrintDot()])
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

# model = build_model()
# # The patience parameter is the amount of epochs to check for improvement
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
# plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} CreditScore".format(mae))

test_predictions = model.predict(normed_test_data).flatten()
error = test_predictions - test_labels
test_predictions = pd.DataFrame(test_predictions)
test_predictions = test_predictions.rename(columns={0: 'Predicted_CreditScore'})
test_labels = pd.DataFrame(test_labels)
df1 = pd.concat([test_dataset,test_labels],axis = 1)
df1.index = test_predictions.index
df2 = pd.concat([df1,test_predictions],axis = 1)
# df2.to_csv('final_pred.csv')
print(df2)