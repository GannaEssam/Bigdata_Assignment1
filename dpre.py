import pandas as pd
from load import dataset, filepath  
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    # Data Cleaning: Replace missing values with mean
    df["Embarked"].fillna(df.Embarked.mode()[0], inplace = True)
    df["Age"].fillna(df.Age.mean(), inplace = True)
    df.drop('Cabin', axis=1, inplace = True)
    df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace = True)

    scaler = MinMaxScaler()
    fare_values = df['Fare'].values.reshape(-1,1)
    scaler.fit(fare_values) 
    df['Fare'] = scaler.transform(fare_values)

    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], dtype = int) 
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    num_bins = 3  # Define the number of bins
    df['Age_Group'] = pd.cut(df['Age'], bins=num_bins, labels=['child', 'adult', 'elderly'])

    num_categories = 4  
    percentiles = [0, 25, 50, 75, 100]  
    fare_labels = ['very_low', 'low', 'medium', 'high']  
    fare_percentiles = df['Fare'].quantile(q=[0, 0.25, 0.5, 0.75, 1])
    df['Fare_Category'] = pd.cut(df['Fare'], bins=fare_percentiles, labels=fare_labels, include_lowest=True)

    return df

def save_preprocessed_data(df, filepath):
    # Save the resulting dataframe
    df.to_csv(filepath, index=False)
    print("Preprocessed data saved as", filepath)

# Usage example: Assume df is the dataset loaded in load.py
data = dataset  # Assuming read_dataset is defined in load.py

preprocessed_df = preprocess_data(data)
print(preprocessed_df.head())
save_preprocessed_data(preprocessed_df, "res_dpre.csv")