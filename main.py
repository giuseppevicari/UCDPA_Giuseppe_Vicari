# DESCRIPTION

# Data Analysis of Diamonds Dataset and application of Machine Learning models for price prediction
# Final Project for Specialist Certificate in Data Analytics - University College Dublin
# Student: Giuseppe Vicari


# INITIALISATION

# Import all libraries used in this project
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error as MSE

# Define random seed for reproducibility
SEED = 42


# ACCESSING AND IMPORTING DATA

# Authenticate via Kaggle API
api = KaggleApi()
api.authenticate()

# Use Kaggle API to download dataset
api.dataset_download_files('shivam2503/diamonds', unzip=True)

# Import the csv file into a Pandas DataFrame
df = pd.read_csv('diamonds.csv')


# EXPLORATORY DATA ANALYSIS & DATA CLEANING

# Define function to explore the structure of a dataframe
def explore_df(dataframe):
    """Prints information on the structure of the dataframe"""
    print("\n Dataframe Shape:", dataframe.shape)
    print('\n')
    print("\n Dataframe Info:")
    print(dataframe.info())
    print('\n')
    print("\n Dataframe Head:")
    print(dataframe.head())
    print('\n')
    print("\n Dataframe Description:")
    print(dataframe.describe().round(2))
    print('\n')


# Analyze structure of diamonds database
explore_df(df)

# Remove unnecessary index column (Unnamed: 0)
df.drop('Unnamed: 0', axis='columns', inplace=True)

# Use regular expressions to count diamonds with lowest clarity grades (SI1, SI2, I1)
my_regex = '\S?I\d'
clar_counter = 0
for clar in df['clarity']:
    if re.match(my_regex, clar):
        clar_counter += 1
print('Count of diamonds with poor Clarity: ', clar_counter)
print('\n')


# TREATMENT OF MISSING DATA

# Count missing values in each column
missing_values_count = df.isnull().sum()
print('\n Missing Values:\n', missing_values_count)
print('\n')

# For every row where any of x/y/z dimensions are zero (effectively missing data), replace with average value
x_mean = np.mean(df['x'])
y_mean = np.mean(df['y'])
z_mean = np.mean(df['z'])

df['x'] = df['x'].replace(0, x_mean)
df['y'] = df['y'].replace(0, y_mean)
df['z'] = df['z'].replace(0, z_mean)


# DATA VISUALIZATION

# Create lists of categorical and numerical features
cat_feat = ['cut', 'color', 'clarity']
num_feat = ['carat', 'depth', 'x', 'y', 'z', 'table']


def plotdf(dataframe):
    """Plots feature distribution and correlation heatmap for given dataframe"""
    # Plot histograms to show distribution of numerical features
    dataframe[num_feat].hist(figsize=(12, 8), bins='auto')

    # Plot distribution of categorical features
    for feat in cat_feat:
        sns.catplot(x=feat, data=dataframe, kind='count', height=3, aspect=1.5)

    # Plot correlation matrix & heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(dataframe.corr(), cmap='Greens', annot=True)
    plt.title('Correlation Heatmap', fontsize=10)
    plt.show()


plotdf(df)

# DATA PRE-PROCESSING

# One-Hot encoding: turn categorical data into a binary vector representation using Pandas get_dummies
df = pd.get_dummies(df, columns=cat_feat)

# Standardize numerical features by removing the mean and scaling to unitary variance using sklearn Standard Scaler
scaler = StandardScaler()
scaled_numerical = pd.DataFrame(scaler.fit_transform(df[num_feat]), columns=num_feat, index=df.index)
df[num_feat] = scaled_numerical[num_feat]

print('Numerical Features after scaling:')
print(df[num_feat].describe().round(2))
print('\n')


# Create Feature Matrix (X) and Target Variable (y) from dataframe
y = df['price']
X = df.drop('price', axis='columns')

print('Shape of X and y dataframes')
print('X', X.shape)
print('y', y.shape)
print('\n')


# SUPERVISED LEARNING

# Create Training and Test sets using a 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Instantiate four regressor models with default values
lreg = LinearRegression()
ridge = Ridge(random_state=SEED)
knr = KNeighborsRegressor()
dtree = DecisionTreeRegressor(random_state=SEED)

# Define a list of tuples containing the classifiers and their respective names
classifiers = [('Linear Regression', lreg), ('Ridge Regressor', ridge), ('K Neighbors Regressor', knr),
               ('Decision Tree Regressor', dtree)
               ]

# Iterating over the list of tuples, perform a 10-fold cross-validation for each regressor model
# For each model, evaluate and print the cross-validation scores (R2 and Root Mean Squared Error)
for classifier_name, classifier in classifiers:
    cv_score_r2 = cross_val_score(classifier, X_train, y_train, scoring='r2', cv=10).mean()
    cv_score_RMSE = -cross_val_score(classifier, X_train, y_train, scoring='neg_root_mean_squared_error', cv=10).mean()

    print('{:s} CV R2 Score: {:.2f} '.format(classifier_name, cv_score_r2))
    print('{:s} CV RMSE Score: {:.2f} '.format(classifier_name, cv_score_RMSE))
    print('\n')

# Perform Adaptive Boosting using Decision Tree regressor as base estimator
# Evaluate R2 and RMSE scores for AdaBoost on the test set
adb_reg = AdaBoostRegressor(base_estimator=dtree, n_estimators=200)  # instantiate regressor
adb_reg.fit(X_train, y_train)  # train model
y_pred = adb_reg.predict(X_test)  # prediction on test set
print('Test Set R2 Score for AdaBoost : {:.2f}'.format(r2_score(y_test, y_pred)))
print('Test Set RMSE Score for AdaBoost : {:.2f}'.format(MSE(y_test, y_pred)**(1/2)))
print('\n')


# Extract test values and AdaBoost residuals into two new datasets
df_testvals = pd.DataFrame({'Test Values': y_test})
df_residuals = pd.DataFrame({'Residuals': y_pred - y_test})  # Residuals: difference between predicted and actual values

# Concatenate dataframes along columns
df_plot = pd.concat([df_testvals, df_residuals], axis=1)
print('Shape of the AdaBoost Residuals dataframe:')
explore_df(df_plot)
print('\n')

# Using the newly created dataframe, plot Residuals for AdaBoost to visually show accuracy of predictions
# The closer to the y=0 line, the better
plt.figure(figsize=(5, 5))
plt.axhline(y=0, color='r', linestyle='dashed')
plt.scatter(df_plot['Test Values'], df_plot['Residuals'], alpha=0.5, s=2)
plt.xlabel('Test Values')
plt.ylabel('Residuals')
plt.show()

# Extract series of feature importances for AdaBoost
importances_adb_reg = pd.Series(adb_reg.feature_importances_, index=X.columns)

# Sort and plot on a bar chart
sorted_importances_adb_reg = importances_adb_reg.sort_values()
sorted_importances_adb_reg.plot(kind='barh', color='blue')
plt.show()


# END
