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
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Karthikeyan R
RegisterNumber:  21222224004

import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1) #removes the specified row or cols
data1.head() 

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear") 
lr.fit(x_train,y_train)
y_pred= lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1= classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
Placement data
![im1](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/adca0381-5056-430e-b5e8-adb637d1500b)

Salary data
![im2](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/ebb15057-994d-4086-b0c8-d6a253040940)

![im3](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/2addd637-9616-4212-9d24-21a2ae86c289)

![im41](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/dc68acee-5991-49fa-8166-18fe73d55aeb)

![im42](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/8ee03e7a-8223-4a6e-b48b-ae2fba9963e9)

![im43](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/7c46b7ae-a68d-4eb2-9dc0-ca9f4651ed3e)

![im44](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/dc3f969b-e5ec-42c2-bd38-f67e7aeccbe7)

![im45](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/dc1aebd6-9a72-4ad5-b7fd-7f76f04bb87e)

![im46](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/c03ccf7b-2663-417b-bc39-9afdb3716222)

![im47](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/e67a40ce-dc3f-4467-8528-045341566e14)

![im48](https://github.com/karthikeyan-R16/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119421232/3e8a2bc9-44e7-4c4f-980c-2f0b85fcb7bb)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
