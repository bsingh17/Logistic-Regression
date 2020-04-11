import numpy as np
import pandas as pd

dataset=pd.read_csv('vehicle_silhouette_dataset.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
lbl_vehicle_class=LabelEncoder()
dataset['vehicle_class']=lbl_vehicle_class.fit_transform(dataset['vehicle_class'])


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=100)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)

print(model.score(x_test,y_test))

import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.xlabel('Range')
plt.ylabel('Predicted Values')
plt.title('Classification')
plt.scatter(range(0,212),y_predict)
plt.show()