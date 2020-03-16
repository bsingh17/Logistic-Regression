import numpy as np 
import pandas as pd

dataset=pd.read_csv('Social_Network_Ads.csv')

from sklearn.preprocessing import LabelEncoder
lbl_gender=LabelEncoder()
dataset.Gender=lbl_gender.fit_transform(dataset['Gender'])

x=dataset.iloc[:,1:4].values
y=dataset['Purchased']

import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='Purchased',data=dataset,palette='hls')
plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)

from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)

y_predict=reg.predict(x_test)
print(reg.score(x,y))
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)