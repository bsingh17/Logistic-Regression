import numpy as np
import pandas as pd

dataset=pd.read_csv('soccer_international_history_dataset.csv')

dataset=dataset.drop(['match_date'],axis='columns')

from sklearn.preprocessing import LabelEncoder
lbl_home_country=LabelEncoder()
dataset['home_country']=lbl_home_country.fit_transform(dataset['home_country'])
lbl_away_country=LabelEncoder()
dataset['away_country']=lbl_away_country.fit_transform(dataset['away_country'])
lbl_match_type=LabelEncoder()
dataset['match_type']=lbl_match_type.fit_transform(dataset['match_type'])
lbl_match_city=LabelEncoder()
dataset['match_city']=lbl_match_city.fit_transform(dataset['match_city'])
lbl_match_country=LabelEncoder()
dataset['match_country']=lbl_match_country.fit_transform(dataset['match_country'])
lbl_result=LabelEncoder()
dataset['home_team_result']=lbl_result.fit_transform(dataset['home_team_result'])

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)

print(confusion)

print(model.score(x_test,y_test))