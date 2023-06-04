# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the data and use label encoder to change all the values to numeric
2. Classify the training data and the test data
3. Calculate the accuracy score, confusion matrix and classification report
4. 

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Karthikeyan R
RegisterNumber:  21222224004

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```
### Output:
![ml4441](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/043cba47-5a28-41d0-84f4-3b1d44a6e3fe)

![ml4442](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/c41ee41a-dc0a-41d5-8a1f-de55be104a86)

![ml4443](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/bb3aa247-1a78-429f-8de5-df97a6afab18)

![ml4444](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/0873d939-1e8d-4ad0-8b6a-7f8f1177e777)

![ml444](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/475b737a-af62-48f8-a751-52b65a6bfd2b)



![ml4 5](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/e92f2e48-a47b-4861-a246-0948f99ea580)


![ml4 6](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/43e894ac-ddbf-4f66-b448-694effa5454f)

![ml4 7](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/3b8b233d-c583-4364-8911-adedd56c055e)

![ml4 8](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/429fb675-4eca-4575-83dd-2e78f1b0b5ce)

![ml4 9](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/931ebd5f-54e4-4e8b-96e0-f3108042ed5d)

![ml4 10](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/124e4a83-8489-4972-9425-e6f03e6fb423)

![ml4 11](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/28a0a036-c5ff-4536-b0ab-0b5bbf431c99)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
