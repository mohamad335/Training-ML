import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset =pd.read_csv("sample_ml_data.csv")
#x is the independent variable and y is the dependent variable
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#fisrt we need to handle the missing data
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
#now we can encode the categorical data
country = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(country.fit_transform(x))
#now we can encode the dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#now we can split the dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#now we can scale the features
#we will scale the features except the first three columns
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
