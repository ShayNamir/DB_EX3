# Implementation of Core Machine Learning Models

This repository features implementations of three essential machine learning algorithms: Naive Bayes, Linear Regression, and Logistic Regression. The code is structured into distinct classes for each algorithm, with a primary script showcasing their application for training and evaluation on various datasets.

## Contents

- [Setup Instructions](#setup-instructions)
- [How to Use](#how-to-use)
- [Overview of Classes](#overview-of-classes)
    - [NaiveBayesModel](#naivebayesmodel)
    - [LinearRegression](#linearregression)
    - [LogisticRegression](#logisticregression)
- [Algorithms and Metrics](#algorithms-and-metrics)
    - [Training and Prediction](#training-and-prediction)
    - [Evaluation Metrics](#evaluation-metrics)
- [Additional Details](#additional-details)

## Setup Instructions

To get started with this project, follow these steps:

1. Clone the repository and navigate to the project directory:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required Python libraries:

    ```bash
    pip install numpy pandas
    ```

   Make sure to replace `<repository_url>` with the URL of your repository and `<repository_directory>` with the name of the directory where the repository is cloned.

## How to Use

1. **Prepare Your Data**: Ensure your data files are in CSV format and located in the same directory as the script.

2. **Execute the Main Script**: Run the script to train and evaluate the models:

    ```bash
    python main.py
    ```

3. **Customization**: Feel free to adjust the `train_test_split` function or model parameters in the script according to your needs.

## Overview of Classes

### NaiveBayesModel

This class implements a basic Naive Bayes classifier. It estimates probabilities based on the frequency of feature values within each class.

#### Key Methods:
- `fit(X, y)`: Trains the model with features `X` and labels `y`.
- `predict(prediction)`: Provides class predictions for new data.
- `score(X_test, y_test)`: Assesses model performance on test data.

### LinearRegression

This class represents a straightforward linear regression model, which fits a linear relationship to the data to predict continuous values.

#### Key Methods:
- `fit(X_train, y_train, iterations=1000, alpha=0.0001)`: Trains the model using gradient descent.
- `predict(X)`: Predicts target values based on input features `X`.
- `mean_squared_error(y_true, y_pred)`: Computes the mean squared error between actual and predicted values.

### LogisticRegression

This class applies logistic regression for binary classification tasks.

#### Key Methods:
- `fit(X, y, iterations=1000, alpha=0.0001)`: Trains the model using gradient descent.
- `predict(X)`: Predicts class labels for the provided features `X`.
- `score(X_test, y_test)`: Evaluates the model on the test set, calculating errors, accuracy, precision, and recall.

## Algorithms and Metrics

### Training and Prediction

- **Naive Bayes**: 
  - Calculates conditional probabilities for each feature given the class.
  - Classifies based on the highest posterior probability.

- **Linear Regression**: 
  - Minimizes the mean squared error between predicted and actual values using gradient descent.

- **Logistic Regression**: 
  - Applies the sigmoid function to convert predictions into probabilities.
  - Uses gradient descent to minimize cross-entropy loss.

### Evaluation Metrics

- **Accuracy**: Measures the proportion of correct predictions.
- **Precision**: The ratio of true positive predictions to all positive predictions.
- **Recall**: The ratio of true positives to all actual positives.
- **F-measure**: The harmonic mean of precision and recall.
- **Mean Squared Error (MSE)**: The average squared difference between actual and predicted values.

---

## Additional Details

- **Data Format**: Ensure datasets are correctly formatted and free from missing values.
- **Customization**: Adjust hyperparameters such as the number of iterations and learning rate as needed.
- **Error Handling**: Consider adding error handling for different data types or missing values.

We welcome contributions and collaborations to enhance this project!
