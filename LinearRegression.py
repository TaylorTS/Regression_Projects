import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#read the data and check on basic information
df = pd.read_csv('FILE PATH')
df.info()
df.columns
df.index
df.head(2)
df.describe().transpose()

#check for any missing data
df.isnull().sum()
#no missing data

#Exploratory Data Analysis
sns.pairplot(data=df, x_vars=['Distance_to_LA','Tot_Bedrooms', 'Median_Income'], y_vars='Median_House_Value', aspect=0.7)
plt.tight_layout()

sns.heatmap(df.corr(), annot=True)

sns.scatterplot(data=df, x='Distance_to_LA', y='Median_House_Value')
sns.scatterplot(data=df, x='Distance_to_SanFrancisco', y='Median_House_Value')
# the further the distance is from LA, the lower the housing price would generally be
# bimodal distribution was detected with distance_to_la--house values are high when the distance_to_la is 0.0 (so when they are very close to LA), and when the distance_to_la is about 0.5 to 0.6 (so when there is certain distance from LA, presumably certain suburban areas)
#similar trend was found with distance_to_SF

#perform train_test_split to train a model
x = df[['Tot_Bedrooms', 'Median_Income']]
y=df['Median_House_Value']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)
lm_pred = lm.predict(x_test)
lm.intercept_
lm.coef_

#evaluate the model
from sklearn import metrics
metrics.mean_absolute_error(y_test, lm_pred)
metrics.mean_squared_error(y_test, lm_pred)
np.sqrt(metrics.mean_squared_error(y_test, lm_pred))
df['Median_House_Value'].mean()

sns.scatterplot(x=y_test, y=lm_pred)
sns.distplot(y_test - lm_pred)

plt.show()
