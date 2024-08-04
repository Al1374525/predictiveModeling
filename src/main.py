import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

#load the data set
california = fetch_california_housing(as_frame=True)
data = california.frame
#Explore the data
print(data.head())
print(data.describe())

#Now we are going to visualize data distribution

sns.histplot(data['MedHouseVal'], bins=30)
plt.title('Distribution of Median House Values')
plt.xlabel('Median House Value ($100,000s)')
plt.ylabel('Frequency')
