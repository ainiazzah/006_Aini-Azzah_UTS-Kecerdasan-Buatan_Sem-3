#Program Single Neuron

# Aini Azzah - 21091397006

#Inisialisasi numpy
import numpy as np

#Inisialisasi variabel
#Input layer feature = 10 
input = [2.0, 1.5, 2.2, 2.5, 3.0, 1.0, 3.1, 1.2, 5.0, 4.5]

#Jumlah Neuron = 1
weight = [-0.1, 0.5, 1.5, -0.2, 0.2, -0.5, 1.0, 1.2, -1.5, 2.0]
bias = 0.5

#Output
output = np.dot(weight, input) + bias

#Print outputnya
print(output)
