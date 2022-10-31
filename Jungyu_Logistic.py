# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 10:35:45 2022

@author: Jungyu Lee, 301236221

Logistic Regression
Exercise #1
"""
import pandas as pd

# a. Get the data
titanic_Jungyu = pd.read_csv('titanic.csv', sep=',')

# b. Initial exploration
    #1. Display the first 3 records.
titanic_Jungyu.head(3)
    #2. Display the shape of the dataframe.
titanic_Jungyu.shape
    #3. Display the names, types and counts (showing missing values per column).
titanic_Jungyu.info()
    #4. (Written Response) From the info identify four columns that are not going to be useful for the model. 
    #5. Display (print the unique values for the following columns : (“Sex”, “Pclass”)
titanic_Jungyu['Sex'].unique()
titanic_Jungyu['Pclass'].unique()

#c. data visualization
    #1. Use pandas crosstab and matplotlib to generate the following diagrams plots.
        #a. A bar chart showing the # of survived versus the passenger class.
import matplotlib.pyplot as plt
pd.crosstab(titanic_Jungyu.Pclass, titanic_Jungyu.Survived)
pd.crosstab(titanic_Jungyu.Pclass, titanic_Jungyu.Survived).plot(kind='bar')
plt.title('# of survived versus Passenger Class_Jungyu')
plt.xlabel('Passenger Class')
plt.ylabel('# of Survived')
        #b. A bar chart showing the # of survived versus the gender.
pd.crosstab(titanic_Jungyu.Sex, titanic_Jungyu.Survived)
pd.crosstab(titanic_Jungyu.Sex, titanic_Jungyu.Survived).plot(kind='bar')
plt.title('# of survived versus Gender_Jungyu')
plt.xlabel('Gender')
plt.ylabel('# of Survived')
        #c. (Written Response) Analyze both plots and write a conclusion from each plot in your written response.
        # The # of survived in the first plot
    #2. (Written Response) Use pandas scatter matrix to plot the relationships between the number of survived a the following features. 
    # (attributes) : Gender, Passenger class, Fare, Number of siblings/spouses aboard, Number of parents/children aboard.
pd.plotting.scatter_matrix(titanic_Jungyu[['Sex', 'Pclass', 'Fare', 'SibSp', 'Parch', 'Survived']])

#d. Data transformation (round #1):    
    #1. Drop the four columns you identified in point (b.4) above.
titanic_Jungyu.drop('PassengerId', axis=1, inplace=True)
titanic_Jungyu.drop('Name', axis=1, inplace=True)
titanic_Jungyu.drop('Ticket', axis=1, inplace=True)
titanic_Jungyu.drop('Cabin', axis=1, inplace=True)
    #2. Using “Get dummies” transform all the categorical variables in your dataframe into numeric values.
    #3. Attach the newly created variables to your dataframe and drop the original columns.
    #4. Remove the original categorical variables columns. Use pandas drop method and select the correct argument values.
titanic_Jungyu = pd.concat([titanic_Jungyu.drop('Sex', axis=1), pd.get_dummies(titanic_Jungyu['Sex'])], axis=1)    
titanic_Jungyu = pd.concat([titanic_Jungyu.drop('Embarked', axis=1), pd.get_dummies(titanic_Jungyu['Embarked'])], axis=1)
titanic_Jungyu = titanic_Jungyu.rename(columns = {'male': 'Sex_Male', 'female': 'Sex_Female', 'C': 'Embarked_C', 'Q': 'Embarked_Q', 'S': 'Embarked_S'})
    #5. Replace the missing values in the Age with the mean of the age.
titanic_Jungyu['Age'].fillna(int(titanic_Jungyu['Age'].mean()), inplace=True)

    #6. Change all column types into float.
titanic_Jungyu = titanic_Jungyu.astype(float)
    #7. By know you should get something like the below when you run pandas info:
titanic_Jungyu.info()

    #8. Write a function that accepts a dataframe as an argument and normalizes all the data points in the dataframe. Use pandas .min() and .max().
def normalize(x):
    return (x-x.min())/(x.max()-x.min())

    #9. Call the new function and pass as an argument your transformed dataframe. By now all your data is numeric.
titanic_Jungyu = normalize(titanic_Jungyu)
    #10. Display (print) the first two records.
titanic_Jungyu.head(2)

    #11. Use pandas.hist to generate a plot showing all the variables histograms. Set the figure size to 9 inches by 10 inches.
titanic_Jungyu.hist(figsize=(9, 10))

    #12. (Written Response) Form the histogram generated focus on the “Port of Embarkation” and write in your written response some highlights.
    #13. Split the features into a dataframe named x_firstname and the target class into another dataframe named y_firstname.
# titanic_Jungyu_final_vars=titanic_Jungyu.columns.values.tolist()
feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_Female', 'Sex_Male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
x_Jungyu = titanic_Jungyu[feature_cols]
y_Jungyu = titanic_Jungyu['Survived']
# x_Jungyu = [i for i in titanic_Jungyu_final_vars if i not in y_Jungyu]

# i. Using Sklearn “train_test_split” split your data into 70% for training and 30% for
# testing, set the random seed to be the last two digits of your student ID number.
from sklearn.model_selection import train_test_split
x_train_Jungyu, x_test_Jungyu, y_train_Jungyu, y_test_Jungyu = train_test_split(x_Jungyu, y_Jungyu, test_size=0.30, random_state=21)
import numpy as np


#e. Build & validate the model
    #1. Using sklearn fit a logistic regression model to the training data.
from sklearn.linear_model import LogisticRegression
Jungyu_model = LogisticRegression()
Jungyu_model.fit(x_train_Jungyu, y_train_Jungyu)

    #2.Display the coefficients (i.e. the weights of the model). 
pd.DataFrame(zip(x_train_Jungyu.columns, np.transpose(Jungyu_model.coef_)))

    #3. (Written Response) Cross validation: 
        #a. Use sklearn cross_val_score to validate the model on the training data.
        #b. Set the number of folds cv to 10.
        #c. Repeat the validation for different splits of the train/test. 
        #   Start at test size 10% and reach test size 50% increasing your test sample by 5%.
        #d. In each run print out the minimum, mean and maximum accuracy of the score.
        #e. Note these results in your writer report and recommended the best split scenario.
from sklearn.model_selection import cross_val_score
from sklearn import linear_model

for i in np.arange (0.1, 0.55, 0.05):
    x_train_Jungyu, x_test_Jungyu, y_train_Jungyu, y_test_Jungyu = train_test_split(x_Jungyu, y_Jungyu, test_size = i, random_state=21)
    LogisticRegression().fit(x_train_Jungyu, y_train_Jungyu)   
    scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'), x_train_Jungyu, y_train_Jungyu, scoring='accuracy', cv=10)
    print('test size -', str(round(i, 2)) + ', min: ', str(scores.min()))
    print('test size -', str(round(i, 2)) + ', mean: ', str(scores.mean()))
    print('test size -', str(round(i, 2)) + ', max: ', str(scores.max()))

#f. Test the model
    #1. Rebuild the model using the 70% - 30% train/test split
x_train_Jungyu, x_test_Jungyu, y_train_Jungyu, y_test_Jungyu = train_test_split(x_Jungyu, y_Jungyu, test_size=0.30, random_state=21)
Jungyu_model = LogisticRegression()
Jungyu_model.fit(x_train_Jungyu, y_train_Jungyu)
    #2. Define a new variable y_pred_Jungyu, store the predicted probabilities of the model in this variable.

from sklearn import metrics
clf1 = linear_model.LogisticRegression(solver='lbfgs')
clf1.fit(x_train_Jungyu, y_train_Jungyu)
y_pred_Jungyu = clf1.predict_proba(x_test_Jungyu)
y_pred_Jungyu
    #3. Define another variable name it y_pred_Jungyu_flag, 
    #   store after transforming the probabilities into a bolean value of true or false based on a threshold value of 0.5. 
y_pred_Jungyu_flag= y_pred_Jungyu[:, 1] > 0.5
y_pred_Jungyu_flag
    #4. From sklearn metrics import : confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
    #5. Print out the accuracy of the model on the test data.

y_predict_Jungyu = clf1.predict(x_test_Jungyu)
y_predict_Jungyu
y_test_Jungyu
print (metrics.accuracy_score(y_test_Jungyu, y_pred_Jungyu_flag))



    #6. Print out the confusion matrix.
confusion_matrix = confusion_matrix(y_test_Jungyu, y_pred_Jungyu_flag)
confusion_matrix
    #7. Print out the classification report.
classification_report(y_test_Jungyu, y_pred_Jungyu_flag)
print(f"{metrics.classification_report(y_test_Jungyu, y_pred_Jungyu_flag)}\n")
    #8. Write down and note the values of : accuracy, precision and re-call
    #9. Repeat steps 3 to 6 with changing the threshold value to 0.75
clf1 = linear_model.LogisticRegression(solver='lbfgs')
clf1.fit(x_train_Jungyu, y_train_Jungyu)
y_pred_Jungyu = clf1.predict_proba(x_test_Jungyu)
y_pred_Jungyu
y_pred_Jungyu_flag= y_pred_Jungyu[:, 1] > 0.75
y_pred_Jungyu_flag
y_predict_Jungyu = clf1.predict(x_test_Jungyu)
y_predict_Jungyu
y_test_Jungyu
print (metrics.accuracy_score(y_test_Jungyu, y_pred_Jungyu_flag))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test_Jungyu, y_pred_Jungyu_flag)
confusion_matrix
classification_report(y_test_Jungyu, y_pred_Jungyu_flag)
print(f"{metrics.classification_report(y_test_Jungyu, y_pred_Jungyu_flag)}\n")
    #10. Compare the accuracy on the test data with the accuracy generated using the training data.
    #11. Compare the values of accuracy, precision and re-call generated at the threshold 0.5 and 0.75. 

