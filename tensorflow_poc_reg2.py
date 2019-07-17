import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
dataframe = pandas.read_csv("lending_club_loans_Clean_MAI_Light_Demo.csv",na_values = ['Missing','NA','na','N/A','n/a',''],encoding = "ISO-8859-1")
dataframe.replace(["NaN", 'NaT','','Missing','NA','na','N/A','n/a','nan','NAN'], numpy.nan, inplace = True)
dataframe.fillna(0,inplace = True)
dataframe.drop(dataframe.columns[[0,1,2]], axis=1, inplace = True)
dataframe[['int_rate','revol_util']]=dataframe[['int_rate','revol_util']].replace('%','',regex=True).astype(float).div(100)

data_num = dataframe._get_numeric_data()
data_cat = dataframe.select_dtypes(exclude=["number"])
data_cat.drop(['initial_list_status','title','pymnt_plan','emp_title', 'issue_d', 'zip_code','earliest_cr_line','last_pymnt_d','next_pymnt_d','last_credit_pull_d','last_credit_pull_d','application_type','addr_state'], axis=1, inplace=True)

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

dataset = df.values
X = dataset[:,1:102]
Y = dataset[:,0]

# define base model
# def baseline_model():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(100, input_dim=100, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(1, kernel_initializer='normal'))
# 	# Compile model
# 	model.compile(loss='mean_squared_error', optimizer='adam')
# 	return model

# define the model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(100, input_dim=100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=larger_model, epochs=100, batch_size=50, verbose=0)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# evaluate model with standardized dataset
# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
