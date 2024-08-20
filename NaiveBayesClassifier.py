import pandas as pd
import numpy as np


class NaiveBayesClassifier:

    def __init__(self):
        self.prior_probabilities = {}
        self.likelihood_probabilities = {}
        self.labels = []
        self.smoothing = 1

    def train(self, X, y):
        self.labels = np.unique(y)
        for label in self.labels:
            self.prior_probabilities[label] = np.sum(y == label) / float(len(y))
            self.likelihood_probabilities[label] = {}
            for feature_index in range(X.shape[1]):
                self.likelihood_probabilities[label][feature_index] = {}
                for feature_value in np.unique(X[:, feature_index]):
                    self.likelihood_probabilities[label][feature_index][feature_value] = (
                        np.sum((X[:, feature_index] == feature_value) & (y == label)) + self.smoothing) / (
                        np.sum(y == label) + self.smoothing * len(np.unique(X[:, feature_index])))

    def classify(self, instances):
        if isinstance(instances, pd.DataFrame):
            instances = instances.values.tolist()
        predictions = []
        for instance in instances:
            class_probabilities = []
            for label in self.labels:
                probability = self.prior_probabilities[label]
                for feature_index in range(len(instance)):
                    feature_value = instance[feature_index]
                    if feature_value in self.likelihood_probabilities[label][feature_index]:
                        probability *= self.likelihood_probabilities[label][feature_value]
                class_probabilities.append(probability)
            predictions.append(self.labels[np.argmax(class_probabilities)])
        return predictions

    def evaluate(self, X_test, y_test):
        metrics = {"True Positive": 0, "True Negative": 0, "False Positive": 0, "False Negative": 0}
        predictions = self.classify(X_test)

        for i in range(len(predictions)):
            if predictions[i] == y_test[i]:
                if predictions[i] == "Yes":
                    metrics["True Positive"] += 1
                else:
                    metrics["True Negative"] += 1
            else:
                if predictions[i] == "Yes":
                    metrics["False Positive"] += 1
                else:
                    metrics["False Negative"] += 1

        accuracy = (metrics["True Positive"] + metrics["True Negative"]) / sum(metrics.values())
        precision = metrics["True Positive"] / (metrics["True Positive"] + metrics["False Positive"]) if (metrics[
                                                                                                           "True Positive"] +
                                                                                                       metrics[
                                                                                                           "False Positive"]) > 0 else 0
        recall = metrics["True Positive"] / (metrics["True Positive"] + metrics["False Negative"]) if (metrics[
                                                                                                        "True Positive"] +
                                                                                                    metrics[
                                                                                                        "False Negative"]) > 0 else 0

        return metrics, accuracy, precision, recall
