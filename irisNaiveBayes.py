# -*- coding: utf-8 -*-
"""
@author: snowfox
"""

import pandas as pd

# Load the dataset
data = pd.read_csv("Iris.csv")

# Split the dataset into features and target variable
X = data.drop('label', axis=1)  # Features
y = data['label']  # Target variable

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Naïve Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Computing TruePositive, FalsePositive, TrueNegative, FalseNegative
TP = cm[1, 1]
FP = cm[0, 1]
TN = cm[0, 0]
FN = cm[1, 0]

accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, average='macro') # macro-averaged precision
recall = recall_score(y_test, y_pred, average='macro') # macro-averaged recall

print("\nMetrics:")
print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)
