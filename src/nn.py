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
    
    def parameters(self):
        return self.weigths + [self.bias]

class Layer:
    def __init__(self, number_of_inputs_to_a_layer, number_of_neurons):
        self.neurons = [Neuron(number_of_inputs_to_a_layer) for _ in range(number_of_neurons)]

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, number_of_inputs, neurons_per_layer):
        layer_size = [number_of_inputs] + neurons_per_layer
        # mlp = MLP(3, [4, 4, 1])
        #layer_size = [3] + [4, 4, 1]  # Resulting in sz = [3, 4, 4, 1]
        self.layers = [Layer(layer_size[i], layer_size[i+1]) for i in range(len(neurons_per_layer))]
         
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]