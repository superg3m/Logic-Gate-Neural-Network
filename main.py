import numpy as np
from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    topology = [6, 8, 8, 6]

    input_data = [
        [1, 0],
        [1, 1],
        [0, 1],
        [1, 1],
        [0, 0],
        [0, 1]]

    expected_output = [0, 1, 0, 1, 0, 0]

    nn = NeuralNetwork(topology, input_data, expected_output,  0.1)
    nn.train()
    while nn.calculate_loss() > 0.01:
        nn = NeuralNetwork(topology, input_data, expected_output, 0.1)
        nn.train()
    nn.print_layers()
    print("Loss: ", nn.calculate_loss())

'''
    learning_rate = 0.1
    epochs = 1000

    nn.train(input_data, expected_output, learning_rate, epochs)
'''
