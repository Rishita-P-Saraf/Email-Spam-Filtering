# Email Spam Filtering

## Overview

This project implements a machine learning-based system to detect spam emails using Python. The system is trained on a dataset of emails labeled as either "spam" or "ham" (non-spam). The goal is to accurately identify and filter out spam emails, enhancing email security and user experience.

## Project Structure

The project consists of the following key components:

*   **Data Loading and Preprocessing**: The dataset is loaded, cleaned, and prepared for model training.
*   **Exploratory Data Analysis (EDA)**: The dataset is explored using visualizations to understand the distribution of spam and ham emails, and common words used in spam emails.
*   **Model Training**: A machine learning model is trained on the preprocessed data.
*   **Model Evaluation**: The trained model is evaluated using various performance metrics.
*   **Spam Detection**: A function is created to predict whether a given email is spam or ham.

## Dataset

The dataset used for this project contains **5572 rows and 5 columns**. The key columns are:
*   **Category**: Indicates whether an email is spam or ham.
*  **Message**: Contains the email text.

The original dataset had some unnamed columns that were dropped because they contained many missing values. The dataset also contained duplicate rows which were removed. The 'v1' and 'v2' columns were renamed to 'Category' and 'Message', respectively. A new column called 'Spam' was created, with values of 1 for spam and 0 for ham.

The dataset has an imbalanced class distribution, with approximately **13.41% spam messages and 86.59% ham messages**.

## Exploratory Data Analysis (EDA)

*   A pie chart visualizes the distribution of spam vs ham messages.
*   A word cloud reveals common words in spam messages, such as **'free', 'call', 'text', 'txt', and 'now'**.

## Model Development

*   The data is split into training and testing sets.
*   A **Multinomial Naive Bayes model** is used for classification.
*   The model is part of a **scikit-learn pipeline** that includes **CountVectorizer** for text vectorization.

## Model Evaluation

The model's performance is evaluated using the `evaluate_model` function, which provides:
*   **ROC AUC score**
*   **Confusion matrix**
*   **Classification report**

The model achieved a **98.49% recall score on the test set**, which shows the model is highly effective at identifying spam emails and minimizing false negatives.

## Spam Detection System

*   The `detect_spam` function takes an email text as input and returns whether the email is predicted to be "This is a Spam Email!" or "This is a Ham Email!".

## Key Insights

*   The Multinomial Naive Bayes model is effective for spam detection.
*   The model has a strong ability to identify spam emails, as indicated by the high recall score.
*   This project demonstrates the effectiveness of machine learning in combating email spam.

## Tech Stack

*   **Python**
*   **Pandas, NumPy**
*   **Scikit-learn**
*   **Matplotlib, Seaborn, WordCloud**

Feel free to fork the project and any suggestions are welcome!
```

