from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

dataset=load_iris()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(dataset.data,dataset.target,test_size=0.2)

from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)

print(reg.score(x_test,y_test))

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(confusion,annot=True)
plt.xlabel('Predictted')
plt.ylabel('True')