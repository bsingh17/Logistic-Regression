import numpy as np
import pandas as pd
dataset=pd.read_csv('hotel booking.csv')

from sklearn.preprocessing import LabelEncoder
lbl_hotel=LabelEncoder()
dataset['hotel']=lbl_hotel.fit_transform(dataset['hotel'])
lbl_arrival=LabelEncoder()
dataset['arrival_date_month']=lbl_arrival.fit_transform(dataset['arrival_date_month'])
lbl_meal=LabelEncoder()
dataset['meal']=lbl_meal.fit_transform(dataset['meal'])
lbl_country=LabelEncoder()
dataset['country']=dataset['country'].astype(str)
dataset['country']=lbl_country.fit_transform(dataset['country'])
lbl_marketsegment=LabelEncoder()
dataset['market_segment']=lbl_marketsegment.fit_transform(dataset['market_segment'])
lbl_distribution=LabelEncoder()
dataset['distribution_channel']=lbl_distribution.fit_transform(dataset['distribution_channel'])
lbl_reserved=LabelEncoder()
dataset['reserved_room_type']=lbl_reserved.fit_transform(dataset['reserved_room_type'])
lbl_assigned=LabelEncoder()
dataset['assigned_room_type']=lbl_assigned.fit_transform(dataset['assigned_room_type'])
lbl_deposit=LabelEncoder()
dataset['deposit_type']=lbl_deposit.fit_transform(dataset['deposit_type'])
lbl_customer=LabelEncoder()
dataset['customer_type']=lbl_customer.fit_transform(dataset['customer_type'])
lbl_reservation=LabelEncoder()
dataset['reservation_status']=lbl_reservation.fit_transform(dataset['reservation_status'])
dataset=dataset.fillna(str(0))
x=dataset.drop('is_canceled',axis='columns')
y=dataset['is_canceled']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)
print(reg.score(x_test,y_test))