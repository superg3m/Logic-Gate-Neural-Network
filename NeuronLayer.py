class NeuronLayer:
    def __init__(self):
        self.neurons = []
        self.final_layer_output = 0

    def linear_combination(self):
        output = 0
        for neuron in self.neurons:
            output += neuron.get_neuron_value()
        self.final_layer_output = output

    def get_final_layer_output(self):
        return self.final_layer_output

    def update_neuron_biases_and_weights(self, learning_rate):
        for neuron in self.neurons:
            neuron.update_bias_and_weight(learning_rate)

    def calculate_gradients(self, error):
        for neuron in self.neurons:
            neuron.calculate_gradient(error)

    def print_layer(self):
        for index, neuron in enumerate(self.neurons):
            print(f"Neuron #{index + 1} | {neuron.get_neuron_value()} | Final layer: {self.get_final_layer_output()}")
