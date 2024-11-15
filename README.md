This project implements a Naive Bayes classifier for three different datasets: breast cancer, car evaluation, and Hayes-Roth. The code is structured similarly for all three datasets, with minor differences in data loading and preprocessing. 

The main components and what the project is about:
# Project Overview :
This project implements a Naive Bayes classifier to perform classification tasks on three different datasets. The main goal is to predict a class label for new instances based on the learned probability distributions from the training data.

Key Components
# Data Loading and Preprocessing:
Each script (NB-Cancer.py, NB-Car.py, NB-hayes-roth.py) has a loadCsv function tailored to its specific dataset.
For the cancer and car datasets, pandas is used to read and preprocess the data, converting categorical variables to numerical codes.
The Hayes-Roth dataset is loaded using the built-in csv module.

# Data Splitting:
The 'splitDataset' function splits the data into training and testing sets.
'crossValSplit' implements k-fold cross-validation.

# Naive Bayes Algorithm Implementation:
'separateByClass': Separates the dataset by class labels.
'summarize and summarizeByClass': Calculate mean and standard deviation for each attribute per class.
'calculateProbability': Implements the Gaussian Probability Density Function.
'calculateClassProbabilities': Calculates the probability of an instance belonging to each class.
'predict': Makes a prediction for a single instance.
'getPredictions': Makes predictions for a set of instances.

# Model Evaluation:
'getAccuracy': Calculates the accuracy of the predictions.

# Cross-Validation and Reporting:
The main execution block implements 2-fold cross-validation.
It reports the accuracy scores, maximum accuracy, and mean accuracy.

Differences Between Scripts
# NB-Cancer.py:
Focuses on breast cancer data with features like age, tumor size, etc.
Uses pandas for data preprocessing.
# NB-Car.py:
Deals with car evaluation data, including features like buying price, maintenance, etc.
Also uses pandas, with specific handling for 'maint' and 'doors' columns.
# NB-hayes-roth.py:
Uses a simpler data loading method with the csv module.
Doesn't require the same level of preprocessing as the other datasets.

Project Purpose
The purpose of this project is to demonstrate the implementation and effectiveness of the Naive Bayes classifier on different types of datasets. It showcases:

• How to implement a Naive Bayes classifier from scratch.

• The versatility of the algorithm across different domains (medical, automotive, and general classification).

• The use of cross-validation for more robust performance estimation.

• Basic data preprocessing techniques for handling categorical data.

By comparing the performance across these datasets, users can gain insights into the strengths and limitations of the Naive Bayes algorithm in different contexts.