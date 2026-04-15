import pandas as pd
import numpy as np

def load_data_safely(file_path):
    """
    Safely load a dataset from a file path.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None
