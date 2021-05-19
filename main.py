# Import packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
import requests
# Import the Excel file into a Pandas DataFrame
df = pd.read_csv('diamonds.csv')

"""
# Web scraping: Retrieve data from the URL using get method
url = "https://www.kaggle.com/shivam2503/diamonds/download"
r = requests.get(url)
df = pd.read_csv(r.content)
"""

# Define random seed for reproducibility
SEED = 42

#Define function to explore the structure of a dataframe
def explore_df(dataframe):
    "Prints information on the structure of the dataframe"
    print("\n Dataframe Shape:", dataframe.shape)
    print("\n Dataframe Info:" )
    print(dataframe.info())
    print("\n Dataframe Head:")
    print(df.head())

explore_df(df)

# DATA CLEANING

# Remove index column "Unnamed: 0"
df.drop('Unnamed: 0', axis='columns', inplace=True)

explore_df(df)
print(df.describe().round(2))

# Check for missing values in each column
missing_values_count = df.isnull().sum()
print('\n Missing Values:\n', missing_values_count)

# Create list of names of categorical features
cat_feat = ['cut', 'color', 'clarity']

# Create Dummies
#df = pd.get_dummies(df, columns=cat_feat)

# Define function to replace categorical features with ordinal features by applying Ordinal Encoder
def cat_to_ord(dataframe, feature_list):
    ord_enc = OrdinalEncoder() #instantiate Ordinal Encoder
    for feature in feature_list: #iterate over feature list
        dataframe[feature+'_Ord'] = ord_enc.fit_transform(dataframe[[feature]]) #create new ordinal feature
        dataframe.drop(feature, axis='columns', inplace=True) #drop categorical feature

# Convert categorical features (Cut, Color, Clarity) to ordinal
cat_to_ord(df, cat_feat)


explore_df(df)

sns_plot = sns.displot(df['price'])
plt.show()