#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from io import StringIO

# Title and Description
st.title("Customer Churn Analysis App")
st.write("This Streamlit app demonstrates customer churn analysis, visualizations, and machine learning.")

# File uploader
#uploaded_file = st.file_uploader("Upload your CSV file (customer_churn.csv)", type=["csv"])

#if uploaded_file is not None:
# Load the dataset
df = pd.read_csv('customer_churn.csv')

# Display a preview of the dataset
st.write("### Dataset Preview")
st.dataframe(df)

# Display basic dataset info
st.write("### Dataset Information")
buffer = StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())  # Output dataset information

# Display summary statistics
st.write("### Summary Statistics")
st.write(df.describe())

# Handle missing or invalid data in 'TotalCharges'
st.write("### Handling Missing Data")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
st.write("Missing values in 'TotalCharges' have been handled.")

# Visualizations
st.write("## Visualizations")

# Select a categorical column for visualization
categorical_features = ['gender', 'SeniorCitizen', "Partner", "Dependents"]
selected_categorical = st.selectbox("Select a categorical column to visualize:", categorical_features)

if selected_categorical:
    st.write(f"### Distribution of {selected_categorical}")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=selected_categorical, data=df, ax=ax)
    ax.set_title(f"Distribution of {selected_categorical}")
    st.pyplot(fig)

# Select a numerical column for a boxplot
st.write("### Boxplot of Monthly Charges by Churn")
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=ax)
ax.set_title("Monthly Charges by Churn")
st.pyplot(fig)

# Advanced visualizations with multiple columns
st.write("### Churn Distribution by Internet Service and Other Features")
advanced_features = ['InternetService', "TechSupport", "OnlineBackup", "Contract"]
for col in advanced_features:
    st.write(f"#### {col}")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Churn", hue=col, data=df, ax=ax)
    st.pyplot(fig)

# Data preprocessing
st.write("## Data Preprocessing")
cat_features = df.drop(['customerID', 'TotalCharges', 'MonthlyCharges', 'SeniorCitizen', 'tenure'], axis=1)
le = preprocessing.LabelEncoder()
df_cat = cat_features.apply(le.fit_transform)

num_features = df[['customerID', 'TotalCharges', 'MonthlyCharges', 'SeniorCitizen', 'tenure']]
finaldf = pd.merge(num_features, df_cat, left_index=True, right_index=True)
st.write("Categorical features encoded and merged with numerical features.")

# Train-Test Split
finaldf = finaldf.dropna()
finaldf = finaldf.drop(['customerID'], axis=1)
X = finaldf.drop(['Churn'], axis=1)
y = finaldf['Churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
st.write("Data split into training and test sets.")

# Handle class imbalance using SMOTE
oversample = SMOTE(k_neighbors=5)
X_train, y_train = oversample.fit_resample(X_train, y_train)
st.write("Class imbalance handled with SMOTE.")

# Model Training
if st.button("Train Random Forest Model"):
    rf = RandomForestClassifier(random_state=46)
    rf.fit(X_train, y_train)

    # Model Evaluation
    preds = rf.predict(X_test)
    accuracy = accuracy_score(preds, y_test)
    st.write(f"### Model Accuracy: {accuracy:.2f}")
# else:
#    st.warning("Please upload the 'customer_churn.csv' file to proceed.")
