import pandas as pd
import numpy as np
from NaiveBayes import NaiveBayesClassifier
from LinearRegression import SimpleLinearRegression
from LogisticRegression import BinaryLogisticRegression


def split_train_test(data, test_ratio=0.2, random_state=53):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(data))
    test_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def display_metrics(metrics, accuracy, precision, recall):
    print(f"Metrics: {metrics}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

################################ Naive Bayes ################################
print("Naive Bayes\n")
naive_bayes_dataset = pd.read_csv('naive_bayes_data.csv')

train_data, test_data = split_train_test(naive_bayes_dataset, 0.8)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

classifier = NaiveBayesClassifier()
classifier.train(X_train, y_train)
metrics, accuracy, precision, recall = classifier.evaluate(X_test, y_test)
display_metrics(metrics, accuracy, precision, recall)

# Example for a new instance
new_instance = pd.DataFrame({'Age': ['young'], 'Income': 'medium'})
print("New instance:\n", new_instance)
print("Prediction for new instance:", classifier.classify(new_instance))


################################ Linear Regression ################################
print("\nLinear Regression\n")

linear_regression_data = pd.read_csv("prices.txt")
train_data, test_data = split_train_test(linear_regression_data, 0.25)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.flatten()
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.flatten()

regressor = SimpleLinearRegression()
regressor.train(X_train, y_train, iterations=1000, alpha=0.0001)
predictions = regressor.predict(X_test)
mse = regressor.calculate_mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

################################ Logistic Regression ################################
print("\nLogistic Regression\n")

logistic_regression_data = pd.read_csv("prices.txt")

train_data, test_data = split_train_test(logistic_regression_data, 0.15, random_state=54)

X_train = train_data.drop(columns=train_data.columns[-2]).values
y_train = train_data.iloc[:, -2].values.flatten()
X_test = test_data.drop(columns=test_data.columns[-2]).values
y_test = test_data.iloc[:, -2].values

classifier = BinaryLogisticRegression()
classifier.train(X_train, y_train, iterations=1000, alpha=0.0001)
predictions = classifier.classify(X_test)
metrics, accuracy, precision, recall = classifier.evaluate(X_test, y_test)
display_metrics(metrics, accuracy, precision, recall)
print("F-measure: ", (2 * precision * recall) / (precision + recall))
