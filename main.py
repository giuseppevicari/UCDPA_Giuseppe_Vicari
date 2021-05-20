# Import packages
import pandas as pd
import numpy as np
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
import requests

"""
# Authenticating via Kaggle API
api = KaggleApi()
api.authenticate()

# Use Kaggle API to download dataset
api.dataset_download_files('shivam2503/diamonds', unzip=True)
"""

# Import the Excel file into a Pandas DataFrame
df = pd.read_csv('diamonds.csv')


# Define random seed for reproducibility
SEED = 2

#Define function to explore the structure of a dataframe
def explore_df(dataframe):
    "Prints information on the structure of the dataframe"
    print("\n Dataframe Shape:", dataframe.shape)
    print("\n Dataframe Info:" )
    print(dataframe.info())
    print("\n Dataframe Head:")
    print(df.head())

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

# Drop rows where any of x/y/z dimensions are zero (effectively missing data)
df.drop(df[df['x'] == 0 ].index , inplace=True)
df.drop(df[df['y'] == 0 ].index , inplace=True)
df.drop(df[df['z'] == 0 ].index , inplace=True)


# Create lists of categorical and numerical features
cat_feat = ['cut', 'color', 'clarity']
num_feat = ['carat', 'depth', 'x', 'y', 'z', 'table']

# Create Dummies
df = pd.get_dummies(df, columns=cat_feat)

"""
# Define function to replace categorical features with ordinal features by applying Ordinal Encoder
def cat_to_ord(dataframe, feature_list):
    ord_enc = OrdinalEncoder() #instantiate Ordinal Encoder
    for feature in feature_list: #iterate over feature list
        dataframe[feature+'_Ord'] = ord_enc.fit_transform(dataframe[[feature]]) #create new ordinal feature
        dataframe.drop(feature, axis='columns', inplace=True) #drop categorical feature

# Convert categorical features (Cut, Color, Clarity) to ordinal
cat_to_ord(df, cat_feat)
"""

scaler = StandardScaler()
scaled_numerical = pd.DataFrame(scaler.fit_transform(df[num_feat]),columns=num_feat,index=df.index)
df[num_feat] = scaled_numerical[num_feat]


print(df.info())

# Create Feature Matrix (X) and Target Variable (y) from dataframe
y = df['price']
X = df.drop('price', axis='columns')

print('X', X.shape)
print('y', y.shape)

"""
#DATA VISUALIZATION
# Plot distribution of values for each feature
X.hist(figsize=(12,8),bins='auto')
plt.figure()

# Plot distribution of values for target
y.hist(figsize=(3,2),bins='fd')
#plt.clf()

# Plot correlation matrix & heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),cmap=plt.cm.Greens,annot=True)
plt.title('Correlation Heatmap', fontsize=10)
plt.show()



# Plot distribution of values for each feature
#X.hist(figsize=(12,8),bins='auto')
#plt.figure()

# Plot distribution of values for target
#y.hist(figsize=(3,2),bins='fd')
"""



# Create Training and Test sets using a 80/20 split. The data set is unbalanced (higher proportion of
# "False" values for IsCancellation target variable), hence I'm using the Stratify option to ensure
# that the random split maintains the correct proportion of True/False values in the train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Instantiate regressor models
lreg = LinearRegression()
#knn = KNN()
#dt = DecisionTreeClassifier(random_state=SEED)
#gnb = GaussianNB()
#lda = LinearDiscriminantAnalysis()

# Define a list of tuples containing the classifiers and their respective names
classifiers = [('Linear Regression', lreg)]

"""
# Iterating over the list of tuples, fit each model to the training set and predict the labels of the test set
# Finally, evaluate and print the accuracy of each model on the test set
for classifier_name, classifier in classifiers:
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print('{:s} : {:.3f}'.format(classifier_name, classifier.score(X_test, y_test)))
    #print('{:s} : {:.3f}'.format(classifier_name, accuracy_score(y_test, y_pred)))
"""



lreg.fit(X_train, y_train)
y_pred = lreg.predict(X_test)
print('Score of Linear Regressor: {:.5f}'.format(lreg.score(X_test, y_test)))

cv_results = cross_val_score(lreg, X, y, cv=5)
print('CV Result of Linear Regressor: {:.2f}'.format(np.mean(cv_results)))


# Instantiate a random forests regressor 'rf' 400 estimators
rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12, random_state=SEED)
# Fit 'rf' to the training set
rf.fit(X_train, y_train)
# Predict the test set labels 'y_pred'
y_pred = rf.predict(X_test)
# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2)
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

# Create a pd.Series of features importances
importances_rf = pd.Series(rf.feature_importances_, index = X.columns)
# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values()
# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()


sys.exit("Testing stop")
