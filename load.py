import sys
import pandas as pd

def read_dataset(filepath):
    try:
        df = pd.read_csv(filepath)  # Assuming the dataset is in CSV format
        return df
    except FileNotFoundError:
        print("File not found.")
        return None

file_path = input("Enter the path to your dataset file: ")
filepath = file_path
dataset = read_dataset(filepath)
print("Dataset successfully loaded:")
print(dataset.head())