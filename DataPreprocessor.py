import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

class DataPreprocessor:
    def __init__(self, data):
        # Create a deep copy of the data to keep the original unchanged
        self.original_data = data
        self.data = data.copy(deep=True)
        self.category_mapping = {}  # This will store the reverse mapping for categorical encodings
                
    def encode_categorical(self):
        categorical_cols = ['product_name', 'product_link', 'shop_name', 'shop_link', 'category']
        for column in categorical_cols:
            self.data[column], uniques = pd.factorize(self.data[column])
            if column == 'category':
                # Ensure 'category' is treated as an integer
                self.data[column] = self.data[column].astype(int)
            self.category_mapping[column] = dict(enumerate(uniques))

    def drop_columns(self):
        # Drop the 'tags' column if it exists
        if 'tags' in self.data.columns:
            self.data.drop('tags', axis=1, inplace=True)

    def tokenize_tags(self):
        # Example tokenization of tag columns (assuming tag columns are from tag_1 to tag_13)
        for i in range(1, 14):  # Adjust range based on your tag columns
            tag_col = f'tag_{i}'
            if tag_col in self.data.columns:
                self.data[tag_col], uniques = pd.factorize(self.data[tag_col])
                self.category_mapping[tag_col] = dict(enumerate(uniques))


    def replace_placeholders(self):
        # Replace 'Please upgrade' with NaN
        self.data.replace('Please upgrade', np.nan, inplace=True)

    def convert_time_strings(self):
        time_cols = ['listing_age', 'shop_age']
        for col in time_cols:
            if self.data[col].dtype != object and self.data[col].isnull().all():
                continue
            self.data[col] = self.data[col].astype(str).str.extract('(\d+)').astype(float)

    def handle_missing_values(self):
        for col in self.data.columns:
            if self.data[col].dtype == 'float64' or self.data[col].dtype == 'int64':
                self.data[col].fillna(self.data[col].mean(), inplace=True)

    def scale_features(self):
        # Check if 'category' is in columns and its data type
        if 'category' in self.data.columns and self.data['category'].dtype in [np.int64, np.float64]:
            numeric_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
            numeric_columns = numeric_columns.drop('category')  # Drop 'category' safely
            self.data[numeric_columns] = StandardScaler().fit_transform(self.data[numeric_columns])
        else:
            numeric_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
            self.data[numeric_columns] = StandardScaler().fit_transform(self.data[numeric_columns])


    def impute_missing_values(self):
        # Impute missing values with mean for numerical columns
        imputer = SimpleImputer(strategy='mean')
        numeric_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data[numeric_columns] = imputer.fit_transform(self.data[numeric_columns])

    def get_preprocessed_data(self):
        return self.data

    def get_original_data(self):
        return self.original_data

    def preprocess(self):
        self.encode_categorical()
        self.drop_columns()
        self.tokenize_tags()
        self.replace_placeholders()
        self.convert_time_strings()
        self.handle_missing_values()
        self.impute_missing_values()  # New method to impute missing values
        self.scale_features()
