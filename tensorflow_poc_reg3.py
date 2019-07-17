import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# load dataset
dataframe = pandas.read_csv("lending_club_loans_Clean_MAI_Light_Demo.csv",na_values = ['Missing','NA','na','N/A','n/a',''],encoding = "ISO-8859-1")
dataframe.replace(["NaN", 'NaT','','Missing','NA','na','N/A','n/a','nan','NAN'], numpy.nan, inplace = True)
dataframe.fillna(0,inplace = True)
dataframe.drop(dataframe.columns[[0,1,2]], axis=1, inplace = True)
dataframe[['int_rate','revol_util']]=dataframe[['int_rate','revol_util']].replace('%','',regex=True).astype(float).div(100)

data_num = dataframe._get_numeric_data()
data_cat = dataframe.select_dtypes(exclude=["number"])
data_cat.drop(['initial_list_status','title','pymnt_plan','emp_title', 'issue_d', 'zip_code','earliest_cr_line','last_pymnt_d','next_pymnt_d','last_credit_pull_d','application_type','addr_state'], axis=1, inplace=True)

#Categorical data preparation:
dummy1 = pandas.get_dummies(data_cat['term'])
del dummy1[' 60 months']
del data_cat['term']
dum_1 = pandas.concat([data_cat, dummy1], axis=1)
# print(dum_1)

dummy2 = pandas.get_dummies(data_cat['grade'])
del dummy2['G']
del dum_1['grade']
dum_2 = pandas.concat([dum_1, dummy2], axis=1)
# print(dum_2)

# print(data_cat['sub_grade'].value_counts())
dummy3 = pandas.get_dummies(data_cat['sub_grade'])
del dummy3['G5']
del dum_2['sub_grade']
dum_3 = pandas.concat([dum_2, dummy3], axis=1)
# print(dum_3)

# print(data_cat['emp_length'].value_counts())
dummy4 = pandas.get_dummies(data_cat['emp_length'])
del dummy4[0]
del dum_3['emp_length']
dum_4 = pandas.concat([dum_3, dummy4], axis=1)
# print(dum_4.columns.values)

# print(data_cat['home_ownership'].value_counts())
dummy5 = pandas.get_dummies(data_cat['home_ownership'])
# print(dummy5)
del dummy5['OWN']
del dum_4['home_ownership']
dum_5 = pandas.concat([dum_4, dummy5], axis=1)
# print(dum_5.columns.values)

# print(data_cat['verification_status'].value_counts())
dummy6 = pandas.get_dummies(data_cat['verification_status'])
# print(dummy6)
del dummy6['Source Verified']
del dum_5['verification_status']
dum_6 = pandas.concat([dum_5, dummy6], axis=1)
# print(dum_6.columns.values)

# print(data_cat['loan_status'].value_counts())
dummy7 = pandas.get_dummies(data_cat['loan_status'])
del dummy7['Late (31-120 days)']
del dum_6['loan_status']
dum_7 = pandas.concat([dum_6, dummy7], axis=1)
# print(dum_7.columns.values)

# print(data_cat['purpose'].value_counts())
dummy8 = pandas.get_dummies(data_cat['purpose'])
del dummy8['renewable_energy']
del dum_7['purpose']
dum_8 = pandas.concat([dum_7, dummy8], axis=1)
# print(dum_8.shape)

df = pandas.concat([data_num,dum_8], axis=1)
# print(df.shape)

# dataset = df.values
# X = dataset[:,1:102]
# Y = dataset[:,0]

train_dataset = df.sample(frac=0.8,random_state=0)
test_dataset = df.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("funded_amnt")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('funded_amnt')
test_labels = test_dataset.pop('funded_amnt')

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
	model = keras.Sequential([
		layers.Dense(100, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
		layers.Dense(100, activation=tf.nn.relu),
		layers.Dense(1)])
	optimizer = tf.keras.optimizers.RMSprop(0.001)
	model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['mean_absolute_error', 'mean_squared_error'])
	return model

class PrintDot(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		if epoch % 100 == 0:
			print('')
			print('.', end='')

EPOCHS = 100
model = build_model()
history = model.fit(normed_train_data, train_labels,epochs=EPOCHS, validation_split = 0.2, verbose=0,callbacks=[PrintDot()])
hist = pandas.DataFrame(history.history)
hist['epoch'] = history.epoch

# model = build_model()
# # The patience parameter is the amount of epochs to check for improvement
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
# plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} funded_amnt".format(mae))

test_predictions = model.predict(normed_test_data).flatten()
error = test_predictions - test_labels
test_predictions = pandas.DataFrame(test_predictions)
test_predictions = test_predictions.rename(columns={0: 'Predicted_funded_amnt'})
test_labels = pandas.DataFrame(test_labels)
df1 = pandas.concat([test_dataset,test_labels],axis = 1)
df1.index = test_predictions.index
df2 = pandas.concat([df1,test_predictions],axis = 1)

# y_true = test_labels.values
# y_pred = test_predictions.values
# test['output_pred'] = model.predict(x=np.array(test[inputs]))
# output_mean = test_labels['funded_amnt'].mean()    # Is this the correct mean value for r2 here?
SSres = numpy.square(test_labels['funded_amnt'] - test_predictions['Predicted_funded_amnt'])
SStot = numpy.square(test_labels['funded_amnt'] - test_labels['funded_amnt'].mean())
r2 = 1 - (SSres.sum()/(SStot.sum()))
print(r2)
# print(test_dataset.columns.values)
# print(pandas.DataFrame(train_labels).columns.values)
# print(test_labels.columns.values)
# df2.to_csv('lending_pred.csv')
# print(df2)