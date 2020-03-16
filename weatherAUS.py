import numpy as np
import pandas as pd

dataset=pd.read_csv('weatherAUS.csv')

#converting the WORDS TO NUMBERS
from sklearn.preprocessing import LabelEncoder
lbl_location=LabelEncoder()
dataset['Location']=lbl_location.fit_transform(dataset['Location'])
lbl_winddir=LabelEncoder()
dataset['WindGustDir']=dataset['WindGustDir'].astype(str)
dataset['WindGustDir']=lbl_winddir.fit_transform(dataset['WindGustDir'])
lbl_winddir9=LabelEncoder()
dataset['WindDir9am']=dataset['WindDir9am'].astype(str)
dataset['WindDir9am']=lbl_winddir9.fit_transform(dataset['WindDir9am'])
lbl_winddir3=LabelEncoder()
dataset['WindDir3pm']=dataset['WindDir3pm'].astype(str)
dataset['WindDir3pm']=lbl_winddir3.fit_transform(dataset['WindDir3pm'])
lbl_raintoday=LabelEncoder()
dataset['RainToday']=dataset['RainToday'].astype(str)
dataset['RainToday']=lbl_raintoday.fit_transform(dataset['RainToday'])
lbl_raintomo=LabelEncoder()
dataset['RainTomorrow']=dataset['RainTomorrow'].astype(str)
dataset['RainTomorrow']=lbl_raintomo.fit_transform(dataset['RainTomorrow'])

dataset=dataset.fillna(str(0))
#now dividing data into training data and testing data
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


#now dividing data into training data and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)

y_predict=reg.predict(x_test)