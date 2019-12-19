import numpy as np


class Perceptron(object):
    # constructor for creating object of class perceptron
    def __init__(self, no_of_inputs, epoch=20, learning_rate=0.01):
        # epoch determines, how many times each training example would pass through perceptron
        self.epoch = epoch
        # learning rate defines that how much change is acceptable between previous and new weights
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        # following line is computing the equation (X1*W1) + (X2*2) + b
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation >= 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.epoch):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                # Following lines are readjusting the weights and bias
                # If the difference is -ve, new weight will be less than previous one
                # If the difference is +ve, new weight will be greater than previous one
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
                print("learning rate", self.learning_rate, self.weights[1:])


training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

labels = np.array([1, 0, 0, 0])

perceptron = Perceptron(2)
perceptron.train(training_inputs, labels)

inputs = np.array([0.5, 0.8])
print(perceptron.predict(inputs))

inputs = np.array([1.5, 0.5])
print(perceptron.predict(inputs))