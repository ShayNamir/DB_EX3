import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.parameters = None

    def train(self, X_train, y_train, epochs=1000, learning_rate=0.0001):
        weights = np.random.rand(len(X_train[0]))
        bias = np.random.rand()
        for _ in range(epochs):
            bias_gradient = np.mean((np.dot(X_train, weights) + bias) - y_train)
            weights_gradient = (1.0 / len(y_train)) * np.dot(((np.dot(X_train, weights) + bias) - y_train), X_train)
            weights -= learning_rate * weights_gradient
            bias -= learning_rate * bias_gradient
        self.parameters = np.concatenate((np.array([bias]), weights))
        print('Model weights:', self.parameters[1:])
        print('Model bias:', self.parameters[0])

    def forecast(self, X):
        return np.dot(X, self.parameters[1:]) + self.parameters[0]

    def compute_mse(self, y_actual, y_predicted):
        for i in range(len(y_actual)):
            print(f'Real price: {y_actual[i]} , Predicted price: {y_predicted[i]}, Error: {y_actual[i] - y_predicted[i]}')
        return np.mean((y_actual - y_predicted) ** 2)
