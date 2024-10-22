# EX2 Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: varsha s
RegisterNumber: 212222220055
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
print('Values of MSE')
```

## Output:
1.df.head()

![image](https://github.com/user-attachments/assets/4bd4db35-07e9-47bd-9acb-c3923e7368b8)


2.df.tail()

![image](https://github.com/user-attachments/assets/2ce1f73f-53dc-481c-9fdd-1076fc3d740d)

3.Array value of X

![image](https://github.com/user-attachments/assets/88261b45-1034-48f9-8950-6823a5b11dd3)

4.Array value of Y

![Array value of Y](https://user-images.githubusercontent.com/128135126/229293494-aa427d62-0d42-4747-9b9d-474c5f58fb29.png)

5.Values of Y prediction

![Values of Y prediction](https://user-images.githubusercontent.com/128135126/229293514-ef09d849-1b86-4783-b366-9552fbafecca.png)

6.Array values of Y test

![Array values of Y test](https://user-images.githubusercontent.com/128135126/229293545-32b41b3c-8494-4138-8f49-6161ed6af60b.png)

7.Training set graph

![Training set graph](https://user-images.githubusercontent.com/128135126/229293565-fbd372b3-aac6-4ed6-be0b-87905f046ebb.png)

8.Test set graph

![Test set graph](https://user-images.githubusercontent.com/128135126/229293588-9a4d9a34-3f38-4e5f-bd77-3e3f79bb2cf0.png)

9.Values of MSE,MAE and RMSE

![Values of MSE,MAE and RMSE](https://user-images.githubusercontent.com/128135126/229293605-f9e791d8-b7c0-45c1-ac1b-2901364e8b26.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
com 
