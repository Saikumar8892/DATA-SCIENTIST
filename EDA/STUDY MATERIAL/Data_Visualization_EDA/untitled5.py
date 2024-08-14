import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
worldmeter = pd.read_csv(r"worldometer_data.csv")
worldmeter.shape
plt.hist(worldmeter.TotalCases)
sns.distplot(worldmeter.TotalCases)
sns.displot(worldmeter.TotalCases)
plt.boxplot(worldmeter.TotalCases)
plt.figure()
sns.kdeplot(worldmeter.TotalCases)
sns.kdeplot(worldmeter.TotalCases, bw=0.5, fill=True)
sns.kdeplot(worldmeter.TotalCases, bw=0.5, fill=False)
