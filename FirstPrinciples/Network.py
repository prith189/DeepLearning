# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:30:51 2018

@author: prithvi
"""

'''
Write a framework to initialize a neural network
Run forward propagation
Run backpropagation and Update weights
'''

import numpy as np

class Sigmoid:
    '''
    Class implementing the Sigmoid activation function
    '''
    
    def __init__(self):
        pass
    
    def activate(self,inputs):
        '''
        Sigmoid activation function on a vector of inputs
        param: inputs - 2D vector of shape (1,num_inputs)
        return: 2D vector of shape (1,num_inputs)
        '''
        return 1. /(1. + np.exp(-1.*inputs))
    
    def derivative(self,inputs):
        '''
        Derivate of a sigmoid activation function on a vector of inputs
        param: inputs - 2D vector of shape (1,num_inputs)
        return: 2D vector of shape (1,num_inputs)
        '''
        return self.activate(inputs)*(1. - self.activate(inputs))    

    
class Linear:
    '''
    Class implementing the Linear activation function
    '''
    def __init__(self):
        pass
    
    def activate(self,inputs):
        '''
        Linear activation function on a vector of inputs
        param: inputs - 2D vector of shape (1,num_inputs)
        return: 2D vector of shape (1,num_inputs)
        '''
        return inputs
    
    def derivative(self,inputs):
        '''
        Derivate of a linear activation function on a vector of inputs
        param: inputs - 2D vector of shape (1,num_inputs)
        return: 2D vector of shape (1,num_inputs)
        '''
        return np.ones_like(inputs)   

    
class Relu:
    '''
    Class implementing the Relu activation function
    '''
    def __init__(self):
        pass
    
    def activate(self,inputs):
        '''
        Relu activation function on a vector of inputs
        param: inputs - 2D vector of shape (1,num_inputs)
        return: 2D vector of shape (1,num_inputs)
        '''
        return np.maximum(0,inputs)
    
    def derivative(self,inputs):
        '''
        Derivate of a Relu activation function on a vector of inputs
        param: inputs - 2D vector of shape (1,num_inputs)
        return: 2D vector of shape (1,num_inputs)
        '''
        d = np.ones_like(inputs)
        d[inputs<=0.] = 0.
        return d   

class Softmax:
    '''
    Class implementing the Softmax activation function
    '''
    def __init__(self):
        pass
    
    def activate(self,inputs):
        '''
        Softmax activation function on a vector of inputs
        param: inputs - 2D vector of shape (1,num_inputs)
        return: 2D vector of shape (1,num_inputs)
        '''
        return np.exp(inputs)/np.sum(np.exp(inputs))
    
    def derivative(self,inputs):
        '''
        Derivative of Softmax activation function on a vector of inputs
        ***NOTE***
        This derivative function is tied up with the CCE_Loss_with_softmax class, so here we are simply returning a ones vector
        **********
        param: inputs - 2D vector of shape (1,num_inputs)
        return: 2D vector of shape (1,num_inputs)
        '''
        return np.ones_like(inputs)  

    
class BCE_loss:
    '''
    Class implementing the Binary Cross entropy loss
    '''
    def __init__(self):
        pass
    
    def compute_loss(self,true,pred):
        '''
        Compute BCE loss between truth and prediction
        param: true - 2D vector of shape (1,1)
        param: pred - 2D vector of shape (1,1)
        return: scalar loss value
        '''
        return (-1)*(true*np.log(pred+1e-11) + (1-true)*np.log(1-pred+1e-11))
    
    def loss_grad(self,true,pred):
        '''
        Compute derivative of BCE loss between truth and prediction of output layer
        param: true - 2D vector of shape (1,1)
        param: pred - 2D vector of shape (1,1)
        return: gradient of loss
        '''
        return -1.*(np.divide(true, pred+1e-11) - np.divide(1 - true, 1 - pred + 1e-11))

    
    

class CCE_loss_for_softmax:
    '''
    Class implementing the Categorical Cross Entropy loss
    '''
    def __init__(self):
        pass
    
    def compute_loss(self,true,pred):
        '''
        Compute CCE loss between truth and prediction
        param: true - 2D vector of shape (1,num_classes)
        param: pred - 2D vector of shape (1,num_classes)
        return: scalar loss value
        '''
        return np.sum((-1)*(true*np.log(pred+1e-11)))
    
    def loss_grad(self,true,pred):
        '''
        Compute CCE loss between truth and prediction as described here: http://cs231n.github.io/neural-networks-case-study/
        param: true - 2D vector of shape (1,num_classes)
        param: pred - 2D vector of shape (1,num_classes)
        return: 2D vector of gradients of shape (1,num_classes)
        '''
        
        return pred-true

    
    
class Layer:
    '''
    Layer class that provides feedforward and backprop functionalities for a single Dense layer
    '''
    def __init__(self,num_input_nodes,num_output_nodes,activation_func):
        '''
        Constructor method that initializes weights/biases/activation function
        param: num_input_nodes - integer value 
        param: num_output_nodes - integer value 
        param: activation_func - string ('relu'/'sigmoid'/'softmax')
        Initializes vectors for weights/biases & weight_updates/bias_updates
        '''
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
        
    
    def forward_prop(self,inputs):
        '''
        Class method that runs a feedforward operation on a set of inputs. Also stores the inputs/pre-activation outputs
        to be used later for backprop computation
        param: inputs - 2D vector of shape (1,num_input_nodes)
        return: activated - 2D vector of shape (1,num_output_nodes)
        '''
        assert (inputs.shape[0] == 1 and inputs.shape[1] == self.num_in)
        self.inputs = inputs
        self.pre_activation= np.matmul(inputs,self.W) + self.B
        activated = self.activation.activate(self.pre_activation)
        return activated
    

    def back_prop(self,errors):
        '''
        Class method that runs a backprop operation on a vector of error gradients received from upstream layers
        param: errors - 2D vector of shape (1,num_output_nodes)
        return: upstream_error - 2D vector of shape (1,num_input_nodes)
        Internally, we compute a 2D array with shape (num_input_nodes,num_output_nodes) which represents the delta weights
        We also compute the upstream error derivative that is passed on to the next layer to aid in its error gradient calculation
        
        Logic:
        The delta weights that are propagated from an output node to each of the input nodes is of the form
        Error at the node * derivative of the activation function w.r.t to pre-activated input * input
        '''
        
        assert (errors.shape[0] == 1 and errors.shape[1] == self.num_out)
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
    
    
    def update_weights(self,lr):
        '''
        Class method to update weights based on a schedule decided by the Network class. 
        Backprop keeps updating the delta_weights until this method is called. Only here are the weight vectors updated
        param: lr - Learning rate to be used
        '''
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
