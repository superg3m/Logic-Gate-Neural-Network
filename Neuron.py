import numpy as np
import math


def tanh_func(value):
    return np.tanh(value)


class Neuron:
    def __init__(self, input_values, bias):  # input_values are a list of ints
        self.input_values = input_values
        self.weight = np.random.randn()
        self.bias = bias
        self.final_value = 0

        self.weight_gradient = 0
        self.bias_gradient = 0

        self.compute_final_value()

    def compute_final_value(self):
        if np.all(self.input_values == 0):
            self.final_value = 0
            return

        if self.bias != 0:
            temp = np.dot(self.input_values, self.weight) + self.bias
            temp = tanh_func(temp)
            self.final_value = temp
        else:
            self.final_value = sum(np.dot(self.input_values, self.weight))

    def update_bias_and_weight(self, learning_rate):
        self.bias = self.bias - (learning_rate * self.bias_gradient)
        self.weight = self.weight - (learning_rate * self.weight_gradient)
        self.compute_final_value()

    def update_neuron_input(self, new_input_value):
        self.input_values = new_input_value
        self.compute_final_value()

    def get_neuron_input(self):
        return self.input_values

    def set_neuron_value(self, new_final_value):
        self.final_value = new_final_value

    def get_neuron_value(self):
        return self.final_value

    def get_neuron_weight(self):
        return self.weight

    def calculate_gradient(self, error):
        # Calculate the gradient with respect to the weights and bias
        weight_gradient = error * self.input_values  # Gradient for weights
        bias_gradient = error  # Gradient for bias

        # Store the gradients for later weight and bias updates
        self.weight_gradient = weight_gradient
        self.bias_gradient = bias_gradient
