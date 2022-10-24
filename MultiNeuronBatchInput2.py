# Aini Azzah - 21091397006 - 2021B
# Program Multi Neuron Batch Input 2 layer

# Inisialisasi numpy
# Input layer feature 10 
# Per batch input = 6
import numpy
inputs = [[4.0, 3.5, 5.0, 2.5, 1.0, 2.2, 2.0, 1.5, 2.0, 3.0],
		  [4.1, 1.0, 2.1, 5.0, 2.0, 3.0, 0.2, 2.7, 1.1, 1.0],
		  [3.0, 1.5, 4.1, 2.2, 0.8, 0.2, 3.5, 2.5, 1.5, 2.0],
		  [2.3, 4.2, 5.5, 3.0, 1.5, 2.5, 0.5, 0.9, 3.7, 1.0],
		  [0.5, 4.5, 3.0, 0.2, 2.5, 2.5, 1.5, 3.5, 5.0, 1.8],
		  [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]]

# Layer 1, Neuron = 5
weights1 = [[-2.1, 3.0, 4.2, 5.1, -1.0, 0.4, -0.6, 2.5, -0.5, 1.7],
		   [4.0, 4.4, -0.2, 1.3, 4.8, 3.1, 0.5, -2.5, 1.0, 5.0],
		   [-0.2,-0.5, 1.7, 0.8, 0.5, 1.0, -1.0, 2.0, 2.0, 2.1],
		   [1.5,-0.1, 0.2, -0.5, 5.0, 3.2, -1.6, 4.1, 2.2, -3.5],
		   [3.0, 4.2,-1.5, 1.5, 1.0, -2.0, 2.5, 2.4, -0.5, -1.0]]
biases1 =   [0.5, 1.0, 2.5, 3.0, 1.5]

# Layer 2, Neuron = 3
weights2 = [[0.1, 1.4, 1.5, 2.0, 1.0],
			[1.5, 1.2, 2.3, 3.1, 2.4],
			[2.4, 3.0, 2.0, 1.1, 1.7]]
biases2 =  [1, 2, 0.5]

# Output layer 1
layer1_outputs = numpy.dot(inputs, numpy.array(weights1).T) + biases1

# Output layer 2
layer2_outputs = numpy.dot(layer1_outputs, numpy.array(weights2).T) + biases2

# Print Output layer 2
print(layer2_outputs) 