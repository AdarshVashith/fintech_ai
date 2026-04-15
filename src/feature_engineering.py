import pandas as pd
import numpy as np

def create_features(df):
    """
    Creates new features for the German Credit dataset and handles missing values 
    and dummy encoding for categorical pipelines.
    """
    df_engineered = df.copy()
    
    # Drop index column if present
    if 'Unnamed: 0' in df_engineered.columns:
        df_engineered = df_engineered.drop(columns=['Unnamed: 0'])
        
    # Deal with typical German Credit Data Categorical string columns
    categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Job']
    
    for col in categorical_cols:
        if col in df_engineered.columns:
            # fillna first
            df_engineered[col] = df_engineered[col].fillna('Unknown').astype(str)
            
    # Dummy encoding
    existing_cat = [c for c in categorical_cols if c in df_engineered.columns]
    if existing_cat:
        df_engineered = pd.get_dummies(df_engineered, columns=existing_cat, drop_first=False)
    
    # For numeric columns
    numeric_fills = ['Age', 'Credit amount', 'Duration']
    for col in numeric_fills:
        if col in df_engineered.columns:
            df_engineered[col] = pd.to_numeric(df_engineered[col], errors='coerce')
            df_engineered[col] = df_engineered[col].fillna(df_engineered[col].median())
            
    return df_engineered
