<H3>NAME: CHARUKESH S</H3>
<H3>REGISTER NO: 212224230044</H3>
<H3>EX. NO.1</H3>
<H3>DATE:25-07-2026</H3>
<H1 ALIGN- =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset from drive
df = pd.read_csv("dataset_2191_sleep.csv");
print(df)
```
```python
#split the dataset
X = df.iloc[:, :-1].values

y = df.iloc[:, -1].values

print(X)
print(y)
```
```python
#Finding Missing Values
print(df.isnull().sum())
```
```python
#Handling Missing values
df.fillna(df.select_dtypes(include='number').mean().round(1), inplace=True)
y = df.iloc[:, -1].values
```
```python
df.drop(['body_weight','max_life_span','total_sleep'], axis=1, inplace = True)
df.info()
```
```python
#Check for Duplicates
df.duplicated()
```
```python
#Detect Outliers
print(df['Lead Streams (in millions)'].describe())
```
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Select only numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Scale only numeric columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(df)
```
```python
#splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

#'test_size = 0.2' means 20% test data and 80% train data
print(X_train)
print(len(X_train))
print(X_test)
print(len(X_test))
```

## OUTPUT:

<img width="760" height="487" alt="image" src="https://github.com/user-attachments/assets/071f2087-0c04-4935-b7a3-c01d8a47731b" />

<img width="527" height="416" alt="image" src="https://github.com/user-attachments/assets/06753e5e-fa76-4550-b910-625bc0c4c9ec" />

<img width="881" height="386" alt="image" src="https://github.com/user-attachments/assets/c0d0cc69-2c7b-4de6-98a0-0969e18519ca" />

<img width="828" height="226" alt="image" src="https://github.com/user-attachments/assets/52679420-a755-42b5-8595-1831ae1cc571" />

<img width="260" height="458" alt="image" src="https://github.com/user-attachments/assets/dbd714b3-9bea-440b-ae7d-74da7581f4e2" />

<img width="1045" height="322" alt="image" src="https://github.com/user-attachments/assets/0fc586ec-a13c-499e-b922-8ca446ab4437" />

<img width="682" height="486" alt="image" src="https://github.com/user-attachments/assets/3deb86b7-3368-42bc-a7c3-e62d3ae8edfa" />

<img width="445" height="495" alt="image" src="https://github.com/user-attachments/assets/86e33bb9-eb69-4011-8b62-a1f75ab4ccfa" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


