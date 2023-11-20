from .network import Network
from .layer import Layer

import numpy as np
    
class ANNBuilder:
    def build(nb_layers, list_nb_nodes, list_functions):
        ann = Network()
        for i in range(0, nb_layers):
            nodes_next_layer = 0
            if i == nb_layers-1:

                nodes_next_layer=1
            else:  
                nodes_next_layer = list_nb_nodes[i+1]

           
            layer = Layer(list_nb_nodes[i],  nodes_next_layer, list_functions[i])
            ann.append(layer)
        return ann


