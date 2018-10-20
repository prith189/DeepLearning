# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:30:51 2018

@author: prithvin
"""

#Write a framework to initialize a neural network
#Run forward propagation
#Run backpropagation and Update weights


import numpy as np

#Sigmoid Activation function
class Sigmoid:
    def __init__(self):
        pass
    
    def activate(self,inputs):
        #Expect to get a 2D array of shape (1,x)
        #Return a 2D array of shape (1,x) by doing element wise sigmoid operation
        return 1. /(1. + np.exp(-1.*inputs))
    
    def derivative(self,inputs):
        #Expect to get a 2D array of shape (1,x)
        #Return a 2D array of shape (1,x) by doing element wise derivative of inputs
        return self.activate(inputs)*(1. - self.activate(inputs))    

    
 
#Linear activation function
class Linear:
    def __init__(self):
        pass
    
    def activate(self,inputs):
        #Expect to get a 2D array of shape (1,x)
        #Return a 2D array of shape (1,x) by doing element wise linear operation
        return inputs
    
    def derivative(self,inputs):
        #Expect to get a 2D array of shape (1,x)
        #Return a 2D array of shape (1,x) by doing element wise derivative of inputs
        return np.ones_like(inputs)   

    
    
#Relu activation function
class Relu:
    def __init__(self):
        pass
    
    def activate(self,inputs):
        #Expect to get a 2D array of shape (1,x)
        #Return a 2D array of shape (1,x) by doing element wise relu operation
        return np.maximum(0,inputs)
    
    def derivative(self,inputs):
        #Expect to get a 2D array of shape (1,x)
        #Return a 2D array of shape (1,x) by doing element wise derivative of inputs
        d = np.ones_like(inputs)
        d[inputs<=0.] = 0.
        return d   

    
    
#Softmax activation function
class Softmax:
    def __init__(self):
        pass
    
    def activate(self,inputs):
        out = np.exp(inputs)/np.sum(np.exp(inputs))
        return out
    
    def derivative(self,inputs):
        #The softmax derivative has been tied up with the Categorial cross entropy loss derivate; See CCE_Loss_for_softmax below
        return np.ones_like(inputs)  

    
    
#Binary Cross Entropy loss for binary classification
class BCE_loss:
    def __init__(self):
        pass
    
    def compute_loss(self,true,pred):
        return (-1)*(true*np.log(pred+1e-11) + (1-true)*np.log(1-pred+1e-11))
    
    def loss_grad(self,true,pred):
        return -1.*(np.divide(true, pred+1e-11) - np.divide(1 - true, 1 - pred + 1e-11))

    
    
#Categorical cross entropy loss for multi class classification to be used with a softmax output layer
class CCE_loss_for_softmax:
    def __init__(self):
        pass
    
    def compute_loss(self,true,pred):
        return (-1)*(true*np.log(pred+1e-11))
    
    def loss_grad(self,true,pred):
        #Reference: http://cs231n.github.io/neural-networks-case-study/
        return pred-true

    
    
#Layer class that provides the single layer feedforward and backprop functionalities
class Layer:
    def __init__(self,num_input_nodes,num_output_nodes,activation_func):
        self.num_in = num_input_nodes
        self.num_out = num_output_nodes
        self.W = np.random.randn(num_input_nodes,num_output_nodes)*0.1
        self.B = np.random.randn(1, num_output_nodes)*0.1
        print(activation_func)
        if(activation_func=='relu'):
            self.activation = Relu()
        if(activation_func=='sigmoid'):
            self.activation = Sigmoid()
        if(activation_func=='softmax'):
            self.activation = Softmax()
        self.delta_weights = np.zeros_like(self.W)
        self.delta_biases = np.zeros_like(self.B)
        self.weight_update_counter = 0
        
    #Method to run the feed forward operation
    def forward_prop(self,inputs):
        #We expect to geta 2D array with shape (1, num_inputs)
        #Will return a 2D array with shape (1, num_outputs)
        assert (inputs.shape[0] == 1 and inputs.shape[1] == self.num_in)
        self.inputs = inputs
        self.pre_activation= np.matmul(inputs,self.W) + self.B
        activated = self.activation.activate(self.pre_activation)
        return activated
    
    #Method to run the backpropagation operation
    def back_prop(self,errors):
        # We expect to get a 2D array with shape (1,num_outputs) as input
        # We compute a 2D array with shape (num_in,num_out) which represents the delta weights
        #We also compute the upstream error derivative that is passed on to the next layer to aid in its error gradient calculation
        assert (errors.shape[0] == 1 and errors.shape[1] == self.num_out)
        #The delta weights that are propagated from an output node to each of the input nodes is of the form
        #Error at the node * derivative of the activation function w.r.t to pre-activated input * input
        error_component = errors*self.activation.derivative(self.pre_activation)
        assert (error_component.shape[0] == 1 and error_component.shape[1] == self.num_out)
        input_component = self.inputs.T
        assert (input_component.shape[0] == self.num_in and input_component.shape[1] == 1)
        del_weights = input_component*error_component
        del_bias = error_component
        upstream_error = np.matmul(self.W, error_component.T).T
        self.delta_weights += del_weights
        self.delta_biases += del_bias
        self.weight_update_counter += 1
        
        return upstream_error
    
    #Update weights whenever the Network class wants to
    def update_weights(self,lr):
        self.W += (-1)*lr*(self.delta_weights/self.weight_update_counter)
        self.B += (-1)*lr*(self.delta_biases/self.weight_update_counter)
        self.delta_weights = np.zeros_like(self.W)
        self.delta_biases = np.zeros_like(self.B)
        self.weight_update_counter = 0
    

    
    
#Network class to build a list of layers and run the train/predict operations
class Network:
    def __init__(self,net_list,lr):
        #Net_list is a list of integers
        #First element is the number of inputs
        #Last element is the number of outputs
        #The elements in between are the number of nodes in the hidden layers
        #For eg, if we plan to run MNIST using 3 hidden layers, then net_list will look like
        #[784, 512, 256, 128, 10]
        self.nn_arch = net_list 
        self.num_layers = len(self.nn_arch) - 1
        self.layers = []
        self.loss = CCE_loss_for_softmax()        
        self.lr = lr
        self.layer_init()
        self.num_sample_proc = 0
        
    
    def layer_init(self):
        for i in range(self.num_layers):
            #If last layer, use sigmoid or softmax, else use Relu
            if(i==self.num_layers-1):
                layer = Layer(self.nn_arch[i],self.nn_arch[i+1],'softmax')
            else:
                layer = Layer(self.nn_arch[i],self.nn_arch[i+1],'relu')
            
            self.layers.append(layer)
                
    def predict(self,inputs):
        #Run the inputs through each layer, compute the class probabilities
        for i in range(len(self.layers)):
            inputs = self.layers[i].forward_prop(inputs)
        return inputs
    
    def train(self,inputs,outputs):
        #Run the inputs through each layer, compute the errors
        computed_output = self.predict(inputs)
        loss = self.loss.compute_loss(outputs,computed_output)
        err_grad = self.loss.loss_grad(outputs,computed_output)
        #Backpropagate the error gradients through the network and compute the del weights and del_biases
        for i in range(len(self.layers)):
            err_grad = self.layers[-(i+1)].back_prop(err_grad)
        self.num_sample_proc += 1
        #Only update weights when certain number of samples have been processed
        if(self.num_sample_proc>3):
            for i in range(len(self.layers)):
                self.layers[i].update_weights(self.lr)
            self.num_sample_proc = 0
        
        return loss
