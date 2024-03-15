import pandas as pd
import numpy as np
from dpre import preprocessed_df

def generate_insights(df):
    insight1 = df.describe()
    
    insight2 = df['SibSp'].value_counts()
    
    insight3 = df[['Survived','Pclass','Age','SibSp','Parch','Fare','Sex_female','Sex_male']].corr()

    insight4  = df["Age_Group"].value_counts()
    
    return insight1, insight2, insight3, insight4

def save_insights(insights):
    i = 1  # Initialize a counter variable
    for insight in insights:
        with open(f"eda-in-{i}.txt", "w") as file:
            file.write(insight.to_string())
        i += 1


save_insights(generate_insights(preprocessed_df))