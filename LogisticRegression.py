import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('FILE PATH')

df.info()
df.index
df.columns
df.head()
df.describe()

#checking and taking care of missing values
df.isnull().sum()

#fill in missing values of glucose with the means by age 
df['glucose'] = df['glucose'].fillna(df.groupby('age')['glucose'].transform('mean'))
df['glucose'].isnull().sum()

#fill in missing values of BMI with the means by age and gender
df[df['BMI'].isnull() == True]
df['BMI'][df['BMI'].isnull() == True]
df['BMI'].groupby([df['age'], df['male']]).mean()
df['BMI'] = df['BMI'].fillna(df.groupby([df['age'], df['male']])['BMI'].transform('mean'))
df['BMI'].isnull().sum()

#drop the column of education
df = df.drop('education', axis=1)

#check on the columns of cigsPerDay. Since the data points that have missing values of cigsPerDay all have currentSmoker = 1, so fill in missing values of cigsPerDay with the mean of cigsPerDay
df['currentSmoker'][df['cigsPerDay'].isnull() == True]
df['cigsPerDay'] = df['cigsPerDay'].fillna(df['cigsPerDay'].mean())

#for BPMeds and heart rate, can't really predict whether the patients are on it or not, so drop the rows that contain missing BPMeds and heartRate
df = df.dropna(subset=['BPMeds', 'heartRate'])

###fit a linear regression model to predict the values of totChol and fill in the missing values of totChol in the dataset with the predicted values###
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
traindf = df[df['totChol'].isnull()==False]
y = traindf['totChol']
traindf_new = traindf.drop('totChol', axis=1)
lr.fit(traindf_new, y)
testdf = df[df['totChol'].isnull()==True]
testdf_new = testdf.drop('totChol', axis=1)
pred = lr.predict(testdf_new)

testdf_new['totChol'] = pred
df['totChol'] = df['totChol'].fillna(testdf_new.totChol)
df.isnull().sum()

#train a logistic model
y_log = df['TenYearCHD']
X_log = df.drop('TenYearCHD', axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
predictions = log_model.predict(X_test)

#evaluate the model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))
