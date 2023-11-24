from .network import Network
from .layer import Layer

import numpy as np
    
class ANNBuilder:
    def build(no_of_layers, list_of_nodes, list_activation_functions):
        ann = Network()
        for i in range(0, no_of_layers):
            nodes_next_layer = 0
            if i == no_of_layers-1:

                nodes_next_layer=1
            else:  
                nodes_next_layer = list_of_nodes[i+1]

           
            layer = Layer(list_of_nodes[i],  nodes_next_layer, list_activation_functions[i])
            ann.append(layer)
        return ann


