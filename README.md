# EX-06-Feature-Transformation

## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

 
# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
## CODE FOR Data_to_Transform.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

df=pd.read_csv("Data_To_Transform.csv")
df

df.skew()

np.log(df["Highly Positive Skew"])

np.reciprocal(df["Moderate Positive Skew"])

np.sqrt(df["Highly Positive Skew"])

np.square(df["Highly Negative Skew"])

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

df["Moderate Positive Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Positive Skew"])
df

df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df

df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])
df

df.skew()

from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew_1'],line='45')
plt.show()

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()

df["Moderate Positive Skew_1"]=qt.fit_transform(df[["Moderate Positive Skew"]])
sm.qqplot(df['Moderate Positive Skew'],line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew_1'],line='45')
plt.show()

df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])
sm.qqplot(df['Highly Positive Skew'],line='45')
plt.show()

sm.qqplot(df['Highly Positive Skew_1'],line='45')
plt.show()

df
# OUPUT
![output](.//h1.png)
![output](.//h2.png)
![output](.//h3.png)
![output](.//h4.png)
![output](.//h5.png)
![output](.//h6.png)
![output](.//h7.png)
![output](.//h8.png)
![output](.//h9.png)
![output](.//h10.png)
## CODE FOR titanic_dataset.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

df=pd.read_csv("titanic_dataset.csv")
df

df.drop("Name",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
df

df.info()

from sklearn.preprocessing import OrdinalEncoder

embark=["C","S","Q"]
emb=OrdinalEncoder(categories=[embark])
df["Embarked"]=emb.fit_transform(df[["Embarked"]])

from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
df

np.log(df["Age"])

np.reciprocal(df["Fare"])

np.sqrt(df["Embarked"])

df["Age _boxcox"], parameters=stats.boxcox(df["Age"])
df

df["Pclass _boxcox"], parameters=stats.boxcox(df["Pclass"])
df

df["Fare _yeojohnson"], parameters=stats.yeojohnson(df["Fare"])
df

df["SibSp _yeojohnson"], parameters=stats.yeojohnson(df["SibSp"])
df

df["Parch _yeojohnson"], parameters=stats.yeojohnson(df["Parch"])
df

df.skew()

from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)


df["Age_1"]=qt.fit_transform(df[["Age"]])
sm.qqplot(df['Age'],line='45')
plt.show()

sm.qqplot(df['Age_1'],line='45')
plt.show()

df["Fare_1"]=qt.fit_transform(df[["Fare"]])
sm.qqplot(df["Fare"],line='45')
plt.show()

sm.qqplot(df['Fare_1'],line='45')
plt.show()

df["Parch_1"]=qt.fit_transform(df[["Parch"]])
sm.qqplot(df["Parch"],line='45')
plt.show()

sm.qqplot(df['Parch_1'],line='45')
plt.show()

df
## output
![output](.//d1.png)
![output](.//d2.png)
![output](.//d3.png)
![output](.//d5.png)
![output](.//d6.png)
![output](.//d7.png)
![output](.//d8.png)
![output](.//d9.png)
### RESULT
The various feature transformation techniques has been performed on the given datasets and the data are saved to a file.