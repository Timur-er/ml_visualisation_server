import numpy as np

# activation function, here we use unit step function
def _unit_step_function(x):
    return np.where(x >= 0, 1, 0)


class Perceptron:

    # initialisation of different parameters
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activation_func = _unit_step_function
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape  # size of input array x
        self.weights = np.zeros(n_features)  # for every sample create 0 weight
        self.bias = 0

        normalised_labels = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iterations):
            for iteration, iteration_value in enumerate(x):
                linear_output = np.dot(iteration_value, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # perceptron update rule
                update = self.learning_rate + (normalised_labels[iteration] - y_predicted)

                self.weights += update * iteration_value
                self.bias += update
            yield self.weights, self.bias  # Возвращает веса и смещение после каждой эпохи

    def predict(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
