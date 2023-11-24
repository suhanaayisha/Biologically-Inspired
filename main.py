

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from activation import *
from loss import *
from ann.annBuilder import ANNBuilder

from adapter.adapter import AdapterFunction

from pso.pso import PSO

dataset = pd.read_csv('datasets/data_banknote_authentication.csv')


rows=dataset.shape[0];
columns=dataset.shape[1];
print("rows = {}".format(rows))
print("columns = {}".format(columns))



X = dataset.iloc[:,:columns-1].values
y = dataset.iloc[:,columns-1:columns].values

sc = StandardScaler()
X = sc.fit_transform(X)


print("y cate", y)

X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2)

no_of_features=X.shape[1];
print("no_of_features = {}".format(no_of_features))




hidden_layers = int(input("Enter number of hiddhen layers (excluding input and output layer): "))




no_of_layers = 2 + hidden_layers
print("Total no. of layers", no_of_layers)





nodes_per_hidden_layer = list(map(int, 
     input("\nEnter the number of nodes in each hidden layers (separated by space).\n For eg. 3 2 1, if you have 3 hidden layers : ").strip().split()))[:hidden_layers]

print("No. of nodes per hidden layer", nodes_per_hidden_layer)

nodes_per_layer = [no_of_features]+nodes_per_hidden_layer+[1]





print("No. of nodes per layer", nodes_per_layer)

sigmoid = Sigmoid()
relu = Relu()
tanh = Tanh()

functions = [relu,sigmoid,tanh]


functions_per_layer_str = list(map( str,
     input("\nEnter the activation functions for each layer (including input and output layer) (separated by space).\n Enter 's' for sigmoid, 'r' for relu and 't' for tan. \n For eg. if there are total of 5 layers s r s t s : ").strip().split()))


 def map_to_activation(s):
     func= sigmoid

    if func == 's': 
        func= sigmoid

    elif func == 'r': 
       func= relu

    elif func == 't': 
         func= tanh        
     return func


 functions_per_layer = map(map_to_activation,functions_per_layer_str)
 functions_per_layer= list(functions_per_layer);
 print("functions_per_layer\n",len(functions_per_layer))

loss_func = Binary_cross_entropy()

 ann = ANNBuilder.build(no_of_layers, nodes_per_layer, functions_per_layer)

annFunc = AdapterFunction(ann, X_train, y_train, loss_func)

#PSO parameters

RANDOM_INFORMANTS = 0
LOCAL_INFORMANTS = 1
LOCAL_GLOBAL_INFORMANTS = 2

size=35
beta=1.3
gamma=1.4
delta=1.3
alpha=0.1
epsilon=0.5
maxIter=100

informantType = RANDOM_INFORMANTS
informantNB = 4
opt = 'MIN'

##PSO population

pso=PSO(annFunc, size, beta, gamma, delta, alpha, epsilon, informantType, informantNB, maxIter, opt)

bestANN = pso.evolve().getANN()

acc, loss =  bestANN.evaluate(X_train, y_train, loss_func)


print("Loss: ",loss)
print("Accuracy: ",acc)
