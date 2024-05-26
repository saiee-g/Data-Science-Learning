# -*- coding: utf-8 -*-
"""
@author: snowfox

Use the inbuilt dataset 'titanic' as used in the above problem. Plot a box plot for
distribution of age with respect to each gender along with the information about whether
theysurvived or not. (Column names : 'sex' and 'age')

"""
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
titanic_data = sns.load_dataset('titanic')

# Filter the dataset
filtered_data = titanic_data[['sex', 'age', 'survived']]

# Plot a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='age', hue='survived', data=filtered_data)
plt.title('Distribution of Age by Gender and Survival Status')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
plt.show()

