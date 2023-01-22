#Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Import the dataset
dataset = pd.read_csv(r"C:\Users\dell\OneDrive\Documents\Data Science\18th\1.POLYNOMIAL REGRESSION\Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#Split the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)"""


'''#Feature Scalling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''


#Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()     
regressor.fit(x, y)


#Predicting a new result
y_pred = regressor.predict([[6.5]])
#now predict previous employee salary & visualize the result


#Plotting the graph
plt.scatter(x, y, color = "red")
plt.plot(x,regressor.predict(x), color = "blue")
plt.title("Truth or bluff (Decision Tree Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
#first part the curve is very good & as I explained this is not a decision tree curve becuase we have to get the tree curve
#algorithm of decission tree is by considering the entrophy and information gain spliting the independent variable into several interval
#as per our tutorial we have 2 independent variable diferent interval forms rectangle & we have to get the average of independent variable that means algorithm will take interval of algorithm
#you have quastion if you are taking average of each interval then how do you have a straight line becuse in decision tree each interval it calculating the average of dependent variable
#and you cannot find the average of independent variable & this is not a continuous regression model & the best way to visualize the non-continuous model
#lets plot the higher resolution using tree models


#if you do advance visualisation along with tree structure then you will get this result only
# Visualising the Decision Tree Regression results (higher resolution)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = "red")
plt.plot(x_grid, regressor.predict(x_grid), color = "blue")
plt.title("Truth or Bluff (Decision Tree Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()


#if you check the plot you found the straight & vertical line here and based on entropy & information gain it splits the whole range in the independent variable to different interval 
#if you check the interval of 6 then you get the point of 150k & the range is 5.5. to 6.5
