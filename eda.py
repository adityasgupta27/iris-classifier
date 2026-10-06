import pandas as pd

df = pd.read_csv("iris.csv")

df = df.drop(columns=["Unnamed: 0"])

df = df.rename(columns={
    "0": "sepal_length",
    "1": "sepal_width",
    "2": "petal_length",
    "3": "petal_width",
    "4": "species"
})

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique values:")
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].unique())

print("\nSummary statistics:")
print(df.describe())


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue="species")
plt.show()