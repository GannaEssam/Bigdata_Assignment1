import pandas as pd
import numpy as np
from dpre import preprocessed_df

import matplotlib.pyplot as plt
def  plot_correlation(data):
    plt.figure(figsize=(8, 6))
    plt.scatter(preprocessed_df['Survived'], preprocessed_df['Fare'], alpha=0.5)
    plt.xlabel('Age')
    plt.ylabel('Fare')
    plt.title('Scatter Plot of Age vs Fare')
    plt.grid(True)
    plt.savefig('vis.png')
    plt.show()
plot_correlation(preprocessed_df)