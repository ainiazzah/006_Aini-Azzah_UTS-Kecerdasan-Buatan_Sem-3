#Program Multli Neuron

# Aini Azzah - 21091397006

#Inisialisasi numpy
import numpy as np

#Inisialisasi variabel
#Input layer feature = 10 
input = [0.5, 1.5, 4.0, 0.2, 3.5, 5.0, 2.5, 3.0, 2.0, 1.0]

#Jumlah Neuron = 5
weight = [[0.2, 0.8, -0.5, 1.0, 2.0, 4.0, 5.0, 2.5, -2.5, -1.0],
          [2.0, -1.5, -1.0, 1.0, -0.5, 5.0, 4.5, 2.5, 3.0, 1.5,],
          [4.2, -0.5, 2.1, 3.0, 5.0, 1.0, 2.0, 3.5, -1.0, -2.0],
          [-1.5, 1.0, -2.4, -3.5, 3.0, 3.5, 1.1, 1.2, 4.5, 5.5],
          [2.2, -1.2, 0.5, 1.0, 0.2, 3.2, 4.4, 3.0, 5.0, -6.0]]

bias = [1.0, 2.5, 2.0, 0.5, 3.0]

#Output
layer_output = np.dot(weight, input) + bias

#Print output
print(layer_output)
