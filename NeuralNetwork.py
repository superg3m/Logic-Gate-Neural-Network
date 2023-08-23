import numpy as np
from Neuron import Neuron
from NeuronLayer import NeuronLayer


class NeuralNetwork:
    def __init__(self, topology, input_data, expected_output, learning_rate):
        self.topology = topology
        self.input_data = input_data
        self.expected_output = expected_output
        self.learning_rate = learning_rate

        self.layers = []

        self.create_input_layers()

        self.create_hidden_layers()

        self.create_output_layer()
        '''
        self.layers[-1].neurons[0].set_neuron_value(0)
        self.layers[-1].neurons[1].set_neuron_value(1)
        self.layers[-1].neurons[2].set_neuron_value(0)
        self.layers[-1].neurons[3].set_neuron_value(1)
        self.layers[-1].neurons[4].set_neuron_value(0)
        self.layers[-1].neurons[5].set_neuron_value(0)
        '''

    def train(self):
        for i in range(1):
            self.forward_propagation()
            #self.backward_propagation()
            #self.update_all_weights_and_biases()



    def create_input_layers(self):
        input_layer = NeuronLayer()
        for i in range(self.topology[0]):
            single_input = self.input_data[i % len(self.input_data)]  # make sure we never go out of bounds
            input_layer.neurons.append(Neuron(single_input, 0))

        input_layer.linear_combination()

        self.layers.append(input_layer)

    def create_hidden_layers(self):
        second_to_last_index = len(self.topology) - 1
        for i in range(1, second_to_last_index):
            hidden_layer = NeuronLayer()
            for j in range(self.topology[i]):
                hidden_layer.neurons.append(Neuron(0, np.random.randn()))

            hidden_layer.linear_combination()
            self.layers.append(hidden_layer)

    def create_output_layer(self):
        output_layer = NeuronLayer()
        for i in range(self.topology[-1]):
            output_layer.neurons.append(Neuron(0, np.random.randn()))

        output_layer.linear_combination()
        self.layers.append(output_layer)

    def calculate_loss(self):
        actual_output = np.array(self.expected_output)

        # Assuming that the output layer neurons hold the final predictions
        predicted_output = np.array([neuron.get_neuron_value() for neuron in self.layers[-1].neurons])

        # Check the shapes of actual and predicted outputs
        if actual_output.shape != predicted_output.shape:
            raise ValueError("Shapes of actual and predicted outputs do not match.")

        mse_loss = np.mean((predicted_output - actual_output) ** 2)
        return mse_loss

    def forward_propagation(self):
        for i in range(1, len(self.layers)):
            for neuron in self.layers[i].neurons:
                neuron.update_neuron_input(self.layers[i - 1].get_final_layer_output())
            self.layers[i].linear_combination()

        return

    def backward_propagation(self):
        output_layer = self.layers[-1]
        error = self.calculate_loss() / len(output_layer.neurons)  # Divide error by the number of neurons
        output_layer.calculate_gradients(error)

        second_to_last_index = len(self.layers) - 2
        for i in range(second_to_last_index, 1, -1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]

            # Calculate the error for the current layer
            error = np.dot(next_layer.neurons[0].weight, next_layer.neurons[
                0].weight_gradient)  # Initialize error with the first neuron's contribution from the next layer

            # Calculate gradients for the current layer
            current_layer.calculate_gradients(error)

    def update_all_weights_and_biases(self):
        for neuron_layer in self.layers:
            neuron_layer.update_neuron_biases_and_weights(self.learning_rate)
        return

    def calculate_cost(self, expected, actual):
        return

    def print_layers(self):
        for index, neuron_layer in enumerate(self.layers):
            print(f"Layer #{index + 1}")
            neuron_layer.print_layer()
            print()
