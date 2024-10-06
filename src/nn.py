from src.backpropagation import Value
import random

class Neuron:
    def __init__(self, number_of_inputs):
        self.weigths = [Value(random.uniform(-1, 1)) for _ in range(number_of_inputs)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # print(list(zip(self.weigths,x)))
        act = sum(wi*xi for wi, xi in zip(self.weigths,x)) + self.bias
        out = act.tanh()
        return out

class Layer:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.neurons = [Neuron(number_of_inputs) for _ in range(number_of_neurons)]

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class MLP:
    def __init__(self, number_of_inputs, number_of_neurons_per_layer):
        sz = [number_of_inputs] + number_of_neurons_per_layer
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(number_of_neurons_per_layer))]
         
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x