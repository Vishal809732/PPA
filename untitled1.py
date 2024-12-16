# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 20:50:47 2024

@author: xaviv
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('movies.csv')

# Selecting relevant features and target
features = ['rating', 'genre', 'year', 'score', 'votes', 'budget', 'runtime']
target = 'gross'

# Dropping rows with missing target values
data_clean = data.dropna(subset=[target])

# Splitting features and target
X = data_clean[features]
y = data_clean[target]

# Preprocessing pipeline for numerical and categorical data
num_features = ['year', 'score', 'votes', 'budget', 'runtime']
cat_features = ['rating', 'genre']

num_transformer = SimpleImputer(strategy='mean')
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# Creating the pipeline with a linear regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")
