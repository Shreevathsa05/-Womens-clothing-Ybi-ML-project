# Title of Project: Sentiment Analysis of Women's Clothing Reviews

# Objective: To build a machine learning model to analyze customer reviews and predict their corresponding ratings.

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

# Import Data
df = pd.read_csv('https://raw.githubusercontent.com/YBIFoundation/ProjectHub-MachineLearning/refs/heads/main/Women%20Clothing%20E-Commerce%20Review.csv')

# Describe Data
df.head()  # Display the first five rows
df.info()  # Get dataset information
df.shape   # Check dimensions of the dataset

# Data Preprocessing
df[df['Review'] == ""] = np.NaN  # Replace empty strings with NaN
df['Review'].fillna("No Review", inplace=True)  # Fill NaN with "No Review"

# Check for missing values
df.isna().sum()

# Extracting features and target
X = df['Review']
y = df['Rating']

# Check value counts for the target variable
df['Rating'].value_counts()

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Text Vectorization
cv = CountVectorizer(lowercase=True, analyzer='word', ngram_range=(2, 3), stop_words='english', max_features=5000)

# Transform training data
X_train = cv.fit_transform(X_train)
cv.get_feature_names_out()  # Check generated features

# Transform test data
X_test = cv.transform(X_test)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model Evaluation
# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# Classification Report
print(classification_report(y_test, y_pred))

# Explanation:
# 1. Preprocessing: Missing reviews were replaced with "No Review". The text data was vectorized into bigram and trigram representations, considering stop words and a maximum of 5000 features.
# 2. Modeling: A Multinomial Naive Bayes model was used for text classification due to its efficiency with text data.
# 3. Evaluation: The confusion matrix and classification report provide insights into the model's performance in predicting customer ratings.
