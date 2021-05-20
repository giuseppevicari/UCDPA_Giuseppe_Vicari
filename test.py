from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
api = KaggleApi()
api.authenticate()

# Download all files of a dataset
# Signature: dataset_download_files(dataset, path=None, force=False, quiet=True, unzip=False)
api.dataset_download_files('shivam2503/diamonds', unzip=True)

# downoad single file
#Signature: dataset_download_file(dataset, file_name, path=None, force=False, quiet=True)
#source = api.dataset_download_file('shivam2503/diamonds','diamonds.csv')
#api.dataset_download_file('shivam2503/diamonds','diamonds.csv', unzip=True)
