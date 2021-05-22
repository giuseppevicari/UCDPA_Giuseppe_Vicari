#### INITIALIZATION ####

# Import packages
import pandas as pd
import numpy as np
import sys
import regex as re
from kaggle.api.kaggle_api_extended import KaggleApi
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error as MSE
#from sklearn.metrics import mean_squared_error as MSE

# Define random seed for reproducibility
SEED = 42

#### ACCESSING AND IMPORTING DATA ####
"""
# Authenticating via Kaggle API
api = KaggleApi()
api.authenticate()

# Use Kaggle API to download dataset
api.dataset_download_files('shivam2503/diamonds', unzip=True)
"""

# Import the Excel file into a Pandas DataFrame
df = pd.read_csv('diamonds.csv')

#### EXPLORATORY DATA ANALYSIS ####

#Define function to explore the structure of a dataframe
def explore_df(dataframe):
    "Prints information on the structure of the dataframe"
    print("\n Dataframe Shape:", dataframe.shape)
    print("\n Dataframe Info:" )
    print(dataframe.info())
    print("\n Dataframe Head:")
    print(dataframe.head())

#explore_df(df)

# DATA CLEANING

# Remove index column "Unnamed: 0"
df.drop('Unnamed: 0', axis='columns', inplace=True)

#explore_df(df)
print(df.describe().round(2))

# TREATMENT OF MISSING DATA
# Check for missing values in each column
missing_values_count = df.isnull().sum()
print('\n Missing Values:\n', missing_values_count)
print('\n')

# Where any of x/y/z dimensions are zero (effectively missing data), replace with average
x_mean = np.mean(df['x'])
y_mean = np.mean(df['y'])
z_mean = np.mean(df['z'])

df['x'] = df['x'].replace(0, x_mean)
df['y'] = df['y'].replace(0, y_mean)
df['z'] = df['z'].replace(0, z_mean)

print(df.describe().round(2))
print('\n')

# Create lists of categorical and numerical features
cat_feat = ['cut', 'color', 'clarity']
num_feat = ['carat', 'depth', 'x', 'y', 'z', 'table']

for feat in cat_feat:
    print(feat, '\n', df[feat].value_counts())
print('\n')

# Use regular expressions to count diamonds with lowest clarity grades (SI1, SI2, I1)
my_regex = '\S?I\d'
clar_counter = 0
for clar in df['clarity']:
    if re.match(my_regex, clar):
        clar_counter +=1
print('Poor clarity: ', clar_counter)
print('\n')

#DATA VISUALIZATION
# Plot distribution of values for each feature
df.hist(figsize=(12,8),bins='auto')

# Plot correlation matrix & heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),cmap=plt.cm.Greens,annot=True)
plt.title('Correlation Heatmap', fontsize=10)
plt.show()


# Create Dummies
df = pd.get_dummies(df, columns=cat_feat)

# Scale Data using sklearn Standard Scaler
scaler = StandardScaler()
scaled_numerical = pd.DataFrame(scaler.fit_transform(df[num_feat]),columns=num_feat,index=df.index)
df[num_feat] = scaled_numerical[num_feat]


# Create Feature Matrix (X) and Target Variable (y) from dataframe
y = df['price']
X = df.drop('price', axis='columns')

print('X', X.shape)
print('y', y.shape)
print('\n')


#### SUPERVISED LEARNING ####

# Create Training and Test sets using a 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Instantiate regressor models
lreg = LinearRegression()
ridge_reg = Ridge()
knr = KNeighborsRegressor()
dtree = DecisionTreeRegressor(random_state=SEED)

# Define a list of tuples containing the classifiers and their respective names
classifiers = [('Linear Regression', lreg), ('Ridge', ridge_reg), ('K Neighbors Regressor', knr),
               ('Decision Tree Regressor', dtree)
               ]

# Iterating over the list of tuples, fit each model to the training set and predict the labels of the test set
# Finally, evaluate and print the R2 and Root Mean Squared Error scores on the test set for each model
for classifier_name, classifier in classifiers:
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cv_score_r2 = cross_val_score(classifier, X_train, y_train, scoring='r2', cv=10).mean()
    cv_score_RMSE = -cross_val_score(classifier, X_train, y_train, scoring='neg_root_mean_squared_error', cv=10).mean()

    print('{:s} CV R2 Score: {:.2f} '.format(classifier_name, cv_score_r2))
    print('{:s} CV RMSE Score: {:.2f} '.format(classifier_name, cv_score_RMSE))
    print('\n')


# Perform AdaBoost
adb_reg = AdaBoostRegressor(base_estimator=dtree, n_estimators=200)
adb_reg.fit(X_train, y_train)
y_pred = adb_reg.predict(X_test)
print('Test Set R2 Score for AdaBoost : {:.2f}'.format(r2_score(y_test, y_pred)))
print('Test Set RMSE Score for AdaBoost : {:.2f}'.format(MSE(y_test, y_pred)**(1/2)))


# create new datasets for plotting
df2 = pd.DataFrame({'Test Values': y_test})
df3 = pd.DataFrame({'Residuals': y_pred - y_test})
#df3 = pd.DataFrame({'Residuals': (y_pred - y_test)/y_pred*100})

# Concatenate dataframes along columns
df4= pd.concat([df2, df3], axis=1)

# Using the newly created dataframe, plot Residuals for AdaBoost to visually show accuracy of predictions
# The closer to the y=0 line, the better
plt.figure(figsize = (5,5))
plt.axhline(y = 0, color = 'r', linestyle = 'dashed')
plt.scatter(df4['Test Values'], df4['Residuals'], alpha=0.5, s=2)
plt.xlabel('Test Values')
plt.ylabel('Residuals')
plt.show()


# Create a Pandas Series of features importances for AdaBoost
importances_adb_reg = pd.Series(adb_reg.feature_importances_, index = X.columns)

# Sort importances
sorted_importances_adb_reg = importances_adb_reg.sort_values()

# Create bar plot of sorted importances
sorted_importances_adb_reg.plot(kind='barh', color='blue'); plt.show()

