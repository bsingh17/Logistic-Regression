import numpy as np
import pandas as pd

dataset=pd.read_csv('german_credit_data_dataset.csv')

from sklearn.preprocessing import LabelEncoder
lbl_check=LabelEncoder()
dataset['checking_account_status']=lbl_check.fit_transform(dataset['checking_account_status'])
lbl_credit=LabelEncoder()
dataset['credit_history']=lbl_credit.fit_transform(dataset['credit_history'])
lbl_purpose=LabelEncoder()
dataset['purpose']=lbl_purpose.fit_transform(dataset['purpose'])
lbl_savings=LabelEncoder()
dataset['savings']=lbl_savings.fit_transform(dataset['savings'])
lbl_present=LabelEncoder()
dataset['present_employment']=lbl_present.fit_transform(dataset['present_employment'])
lbl_personal=LabelEncoder()
dataset['personal']=lbl_personal.fit_transform(dataset['personal'])
lbl_otherdebtors=LabelEncoder()
dataset['other_debtors']=lbl_otherdebtors.fit_transform(dataset['other_debtors'])
lbl_property=LabelEncoder()
dataset['property']=lbl_property.fit_transform(dataset['property'])
lbl_otherinstallment=LabelEncoder()
dataset['other_installment_plans']=dataset['other_installment_plans'].astype(str)
dataset['other_installment_plans']=lbl_otherinstallment.fit_transform(dataset['other_installment_plans'])
lbl_housing=LabelEncoder()
dataset['housing']=lbl_housing.fit_transform(dataset['housing'])
lbl_job=LabelEncoder()
dataset['job']=lbl_job.fit_transform(dataset['job'])
lbl_telephone=LabelEncoder()
dataset['telephone']=lbl_telephone.fit_transform(dataset['telephone'])
lbl_foreign=LabelEncoder()
dataset['foreign_worker']=lbl_foreign.fit_transform(dataset['foreign_worker'])

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)



from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)

print(reg.score(x_test,y_test))