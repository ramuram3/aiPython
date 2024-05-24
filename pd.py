import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
irisDf = pd.read_csv('iris.csv')
# print(irisDf.info())
# print(irisDf.describe())
# print(irisDf['species'].value_counts())
# print(irisDf[irisDf.duplicated()])
refinedIrisDf = irisDf.drop_duplicates()
# print(refinedIrisDf)
# grouped = refinedIrisDf.groupby('species')
# print(grouped.sum())
grouped = refinedIrisDf.groupby('species')['sepal_length'].mean().reset_index()

fig, axs = plt.subplots(3, 2)
#sns.countplot(x='species',data=refinedIrisDf, ax=axs[0,0])
axs[0,0].bar(grouped['species'],grouped['sepal_length'])
axs[0,1].hist(refinedIrisDf['sepal_length'])
axs[0,1].set_title('sepal_length')
axs[1,0].hist(refinedIrisDf['sepal_width'])
axs[1,0].set_title('sepal_width')
axs[1,1].hist(refinedIrisDf['petal_length'])
axs[1,1].set_title('petal_length')
axs[2,0].hist(refinedIrisDf['petal_width'])
axs[2,0].set_title('petal_width')
sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=refinedIrisDf, ax=axs[2, 1])
axs[2, 1].set_title('Scatter Plot by Seaborn')
plt.show()

