import numpy as np
import pandas as pd

class LogisticClassifier:
    def activation_function(self, z):
        return 1 / (1 + np.exp(-z))

    def hypothesis(self, X, weights, bias):
        return self.activation_function(np.dot(X, weights) + bias)

    def train(self, X, y, epochs=1000, learning_rate=0.0001):
        self.weights = np.random.rand(len(X[0]))
        self.bias = np.random.rand()
        learning_rate = 0.001
        for _ in range(10000):
            bias_gradient = np.mean(self.hypothesis(X, self.weights, self.bias) - y)
            weights_gradient = (1.0 / len(y)) * np.dot((self.hypothesis(X, self.weights, self.bias) - y), X)
            self.bias -= learning_rate * bias_gradient
            self.weights -= learning_rate * weights_gradient

        print('Model weights:', self.weights)
        print('Model bias:', self.bias)

    def classify(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = self.activation_function(linear_output)
        return [1 if i > 0.5 else 0 for i in predictions]

    def evaluate(self, X_test, y_test):
        results = {"True Positive": 0, "True Negative": 0, "False Positive": 0, "False Negative": 0}
        predicted_classes = self.classify(X_test)

        for i in range(len(predicted_classes)):
            if predicted_classes[i] == y_test[i]:
                if predicted_classes[i] == 1:
                    results["True Positive"] += 1
                else:
                    results["True Negative"] += 1
            else:
                if predicted_classes[i] == 1:
                    results["False Positive"] += 1
                else:
                    results["False Negative"] += 1

        accuracy = (results["True Positive"] + results["True Negative"]) / sum(results.values())
        precision = results["True Positive"] / (results["True Positive"] + results["False Positive"]) if (results["True Positive"] + results["False Positive"]) > 0 else 0
        recall = results["True Positive"] / (results["True Positive"] + results["False Negative"]) if (results["True Positive"] + results["False Negative"]) > 0 else 0
        return results, accuracy, precision, recall
