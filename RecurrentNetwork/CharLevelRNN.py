# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:25:30 2018

@author: prithvi
"""

'''
Build a LSTM based recurrent network for predicting the next character in a sequence, and run it on a Python/C codebase to generate
computer code-like text.
'''

import numpy as np
import random
from keras.layers import Input, LSTM, Dense
from keras.models import Model
import os
from sklearn.preprocessing import OneHotEncoder

def sample(preds,eps):
    '''
    Utility function that takes the output probability (ie. the output of a softmax layer), and picks the best class as the prediction
    In the normal case, we would do this as np.argmax(preds), but for text generation we would need to add some randomness,
    in order to not predict the same text over and over again.
    The logic used here was borrowed from Reinforcement learning called Epsilon greedy search.
    where eps = 1.0 equates to using np.argmax and eps = 0.0 equates to random picking irrespective of the values in the preds vector.
    param: preds - Vector of probabilities
    param: eps - Epsilon for the epsilon greedy search algorithm
    '''
    num_classes = np.shape(preds)[0]
    thresh_prob = 1./num_classes
    num_true_classes = num_classes - np.sum(preds<=thresh_prob)
    eps_p = eps/num_true_classes
    max_idx = np.argmax(preds)
    preds[preds<=thresh_prob] = 0.0
    preds[preds>=thresh_prob] = eps_p
    preds[max_idx] = 1 - eps + eps_p
    return preds

class LSTMNetwork:
    '''
    Class that builds, trains a Single layer LSTM network
    '''
    def __init__(self,timesteps,num_features,num_classes):
        '''
        Constructor method
        param: timesteps - Number of timesteps in the input sequence
        param: num_features - Number of features for each element of the input sequence. For one hot encoded characters, this
        will be equal to the number of classes
        param: num_classes - Number of classes to define the size of the output layer
        '''
        self.ts = timesteps
        self.nf = num_features
        self.nc = num_classes
    
    def build_net(self):
        '''
        Function that defines and compiles a network with a single LSTM layer. Uses Adam optimizer.
        '''
        inp = Input([self.ts,self.nf])
        x = LSTM(128, return_sequences = True)(inp)
        y = Dense(self.nc,activation='softmax')(x)
        self.model = Model(inputs=inp, outputs=y)        
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        
    def train(self, data_generator, num_epochs, steps_per_epoch):
        '''
        Function to train the network
        param: data_generator - A generator that yields the feature vector of size: [batch_size, num_timesteps, num_classes]
        and a target vector also of the same size
        param: num_epochs - Number of epochs to train
        param: steps_per_epoch - Number of iterations before an epoch is declared complete
        '''
        self.model.fit_generator(generator = data_generator,
                                   steps_per_epoch = steps_per_epoch,
                                   epochs = num_epochs)
        pass
    
    def predict(self, input_sample):
        '''
        Function to predict on a new sample
        param: input_sample - A feature vector of size: [batch_size, num_timesteps, num_classes]
        '''
        return self.model.predict(input_sample)


class text_gen:
    def __init__(self,text_file_dir,num_timesteps,num_samples):
        '''
        Constructor method for the text generator
        param: text_file_dir - Directory where input text files are saved
        param: num_timesteps - How many characters do we want the network to keep a history of
        param: num_samples - batch_size
        '''
        self.file_dir = text_file_dir
        self.files_in_dir = os.listdir(self.file_dir)
        self.ts = num_timesteps
        self.ns = num_samples
        self.char_list = []
        self.read_file()
        self.ohe = OneHotEncoder(n_values=self.num_classes,sparse=False)
        
    def read_file(self):
        '''
        Function to read text files from a directory, and split them into characters. Converts from characters to integers, and
        finally stores all the character sequences as a list of integers (self.int_list). Downstream functions will 
        use self.int_list to sample and feed to network
        '''
        for fi in self.files_in_dir:
            file_to_read = self.file_dir + '/' + fi
            f = open(file_to_read,'r')
            self.char_list.extend(list(f.read().lower()))
            f.close()
        self.len_char_list = len(self.char_list)
        self.unique_char = list(set(self.char_list))
        self.num_classes = len(self.unique_char)
        self.int_mapping()
        self.int_list = [self.char_to_int[i] for i in self.char_list]
    
    def int_mapping(self):
        '''
        Function to setup dictionaries to convert from character to integer/integer to character. These are needed to convert from 
        text sequences to integer sequences that can be fed to a network
        '''
        self.char_to_int = {}
        self.int_to_char = {}
        for i in range(self.num_classes):
            self.char_to_int[self.unique_char[i]] = i
            self.int_to_char[i] = self.unique_char[i]
    
    def reshape_int_to_vector(self,arr):
        '''
        Utility function to convert from integer to one hot vector
        param: arr - A single integer in numpy array of shape (1,1)
        '''
        return self.ohe.fit_transform(np.expand_dims(np.array(arr),axis=1))
        
    def get_sequence(self):
        '''
        Sample a short sequence and corresponding target vector. Eg. if sample if 'excess bagg', then target will be 'xcess bagga'
        '''
        start = random.randint(0,self.len_char_list - self.ts - 2)
        data = self.int_list[start : start+self.ts]
        target = self.int_list[start+1 : start+self.ts+1]
        data = self.reshape_int_to_vector(data)
        target = self.reshape_int_to_vector(target)
        return data, target
    
    def predict_text(self, nnet, seed_char, num_char):
        '''
        Function that uses a trained network to generate new text
        param: nnet - Trained neural network that has a .predict() method
        param: seed_char - The seed text to use
        param: num_char - Number of characters of text to generate
        '''
        
        #Truncate the seed text to self.ts
        if(len(seed_char)>self.ts):
            seed_char = seed_char[-self.ts:]
        
        #Run the network on the seed text to bring the hidden states to intended place
        count = 0
        inp_arr = np.zeros([1,self.ts,self.num_classes])
        for i in range(len(seed_char)):
            nnet.model.reset_states()
            inp_arr[0,count,self.char_to_int[seed_char[i]]] = 1
            pred_char_int = np.random.choice(self.num_classes,1,p=nnet.predict(inp_arr)[0][count])[0]
            count += 1            
        
        #Feed the output as the next input character and run the network until num_char are generated
        pred_text = seed_char
        for i in range(num_char):
            nnet.model.reset_states()
            if(count>=self.ts):
                inp_arr = np.roll(inp_arr,-1,axis=1)
                count -= 1
                inp_arr[0,count,:] = np.zeros([self.num_classes])
            inp_arr[0,count,pred_char_int] = 1
            pred_prob = sample(nnet.predict(inp_arr)[0][count],0.1)
            pred_char_int = np.random.choice(self.num_classes,1,p=pred_prob)[0]
            pred_text = pred_text + self.int_to_char[pred_char_int]
            count += 1
            
        return pred_text
    
    def gen(self):
        '''
        Generator that yields training batches of size [batch_size, num_timesteps, num_classes]
        '''
        while(1):
            data = np.zeros([self.ns,self.ts,self.num_classes])
            target = np.zeros([self.ns,self.ts,self.num_classes])
            for i in range(self.ns):
                data[i],target[i] = self.get_sequence()
            yield data, target
        
    def get_seed_text(self,len_seed):
        '''
        Utility function to generate a random seed text for the prediction phase
        param: len_seed - Length of the seed to generate
        '''
        t_gen = self.gen()
        seed_text,_ = next(t_gen)
        seed_text = np.argmax(seed_text[0],axis=1)
        seed_text = [self.int_to_char[i] for i in seed_text]
        seed_text = ''.join(seed_text)
        seed_text = seed_text[:len_seed]
        return seed_text
    
if __name__ == '__main__':
    '''
    Demonstrate the usage of the defined classes. Train the network and show generated text
    '''
    num_timesteps = 256
    batch_size = 64
    text_file_dir = 'Text_Files_C'
    t_gen = text_gen(text_file_dir,num_timesteps,batch_size)
    net = LSTMNetwork(num_timesteps,t_gen.num_classes,t_gen.num_classes)
    net.build_net()
    batch_gen = t_gen.gen()
    for i in range(10):
        net.train(batch_gen,10,1000)
        print('Generated Text after Epoch: ',i)
        pred_text = tg.predict_text(net,t_gen.get_seed_text(),800)
        pred_text = ''.join(pred_text)
        print(pred_text)

