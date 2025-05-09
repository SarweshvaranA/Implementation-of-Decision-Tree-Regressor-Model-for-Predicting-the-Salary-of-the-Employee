# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries (pandas, sklearn modules).
2. Read the CSV file into a DataFrame.
3. Show the first few rows with head().
4. Display info about data types and null values with info().
5. Check for missing values with isnull().sum().
6. Encode the "Position" column using LabelEncoder.
7. Set features x as "Position" and "Level", target y as "Salary".
8. Split the data into training and testing sets (80%-20%).
9. Create and train a DecisionTreeRegressor model.
10. Predict salaries on the test set.
11. Calculate Mean Squared Error (MSE).
12. Calculate R² Score for accuracy.
13. Predict salary for a new input [5,6].

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SARWESHVARAN A
RegisterNumber:  212223230198
*/
```
```python
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
```

![image](https://github.com/user-attachments/assets/2afa0cb6-9ffa-407d-860e-d7b07569d377)

```python
data.info()
```
![image](https://github.com/user-attachments/assets/afa093b3-3044-4df3-a153-f9293c345e7c)

```python
data.isnull().sum()
```

![image](https://github.com/user-attachments/assets/dca0eec3-8f7f-4720-a1ef-1142b143501e)

```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
```

![image](https://github.com/user-attachments/assets/a330a668-2e8d-4dd0-aab1-9cf86d61267a)

```python
x=data[["Position","Level"]]
x.head()
```
![image](https://github.com/user-attachments/assets/d426a3ff-e843-4b90-8b2d-68a8380a4091)


```python
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
```

![image](https://github.com/user-attachments/assets/5b723790-dbae-43ef-baad-2a75b197ed99)

```python
r2=metrics.r2_score(y_test,y_pred)
r2

```
![image](https://github.com/user-attachments/assets/92e63bb5-da66-45f8-ae8a-5a0e06896a0b)

```python
dt.predict([[5,6]])
```
![image](https://github.com/user-attachments/assets/c117cdc2-8db2-458b-9975-1cc42739e895)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
