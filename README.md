# Implementation of Simple Linear Regression Model for Predicting the Marks Scored

## AIM
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages
2. Assigning hours To X and Scores to Y
3. Plot the scatter plot
4. Use mse,rmse,mae formmula to find

## Program
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Naresh kumar
RegisterNumber:  212220223005

```
```

import pandas as pd
import numpy as np
df=pd.read_csv('student_scores.csv')
print(df)
df.head()
df.tail()
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
print(X,Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,reg.predict(X_test),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output
### Initial Dataframe


![out](14.png)

### df.head()


![out](15.png)

### df.tail()



![out](16.png)

![output](3.png)


![output](6.png)

![out](9.png)

![out](11.png)

### The values of MSE , MAE and RMSE


![out](13.png)


## Result
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
