# -*- coding: utf-8 -*-
"""
@author: snowfox

Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains
information about the passengers who boarded the unfortunate Titanic ship. Use the
Seaborn library to seeif we can find any patterns in the data.
2. Write a code to check how the price of the ticket (column name: 'fare') for each
passengeris distributed by plotting a histogram.

"""
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
titanic_data = sns.load_dataset('titanic')

# Explore the dataset
print(titanic_data.head())  # Display the first few rows of the dataset
print(titanic_data.info())  # Display information about the dataset, including column names and data types

# Plot a histogram of ticket prices
sns.histplot(titanic_data['fare'], kde=True)
plt.title('Distribution of Ticket Prices')
plt.xlabel('Ticket Fare')
plt.ylabel('Frequency')
plt.show()
