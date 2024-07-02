## AEROFIT BUSINESS CASE

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


aerofit = pd.read_csv("C:\\Users\\DELL\\Downloads\\Aerofit.txt") 

print(aerofit.head(10))
print(aerofit.shape)
print(aerofit.dtypes)
print(aerofit.isnull().sum())
print(aerofit.duplicated().sum())
print(aerofit.nunique())
print(aerofit.describe(include='all'))

## Univariate Analysis - Outlier Detection

d = ['Age', 'Education', 'Usage', 'Income', 'Fitness', 'Miles']
data = aerofit[d]

for col in data:
    sns.boxplot(data=aerofit[col])
    plt.title(f"Outliers for {col}")
    plt.show()


## Bivariate Analysis – Checking if Gender or Marital Status have any effect on product purchased.
sns.countplot(data=aerofit, x='Product', hue='Gender', palette='Set2') 
plt.xlabel("Product") 
plt.ylabel("Count") 
plt.title("Product vs Gender") 
plt.show()

sns.countplot(data=aerofit, x='Product', hue='MaritalStatus', palette='Set3') 
plt.xlabel("Product") 
plt.ylabel("Count") 
plt.title("Product vs Marital Status") 
plt.show()

    ## Checking if the following Age,Education, Usage, Fitness, Income, Miles have any effect on the product purchased.

for col in data:
    sns.boxplot(x=aerofit['Product'], y=aerofit[col], palette= 'Set3')
    plt.title(f"Product vs {col}")
    plt.show()


## Multivariate Analysis - Visualising the Relationship between Numerical Veriables.

sns.pairplot(aerofit, hue='Product', diag_kind='kde') 
plt.show()

for col in data:
    sns.boxplot(data=aerofit, x='Gender', y=data[col], hue='Product', palette='Set3')
    plt.title(f"Product vs {col}")
    plt.legend(loc = 'upper left')
    plt.show()


## Marginal Probability – Using Crosstab.
cr = ['Gender', 'MaritalStatus', 'Fitness', 'Usage']
cross = aerofit[cr]

for col in cross:
    print(pd.crosstab(index=aerofit[col],columns=aerofit['Product'], normalize=True))
    print()


## Correlation between different features
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True,cmap='coolwarm', fmt=".2f")
plt.show()