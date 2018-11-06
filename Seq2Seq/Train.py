# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 21:37:36 2018

@author: prithvi
"""

import numpy as np
import random
from keras.layers import Input, LSTM, Dense, Embedding
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder

class LSTMSeq2Seq:
    #Building an LSTM network to generate sequences from input sequences
    #This encoder - decoder architecture can be expanded to multiple use cases
    #Here we will look at a simple sorting use case
    def __init__(self, num_classes, len_max_seq, embed_dim):
        self.num_classes = num_classes
        self.len_max_seq = len_max_seq
        self.embed_dim = embed_dim
        self.build_net()
    
    def build_net(self):
        enc_input = Input([self.len_max_seq,])
        enc_embed = Embedding(self.num_classes,self.embed_dim)(enc_input)
        enc_output, hs, cs = LSTM(50,return_state=True)(enc_embed)
        
        dec_input = Input([self.len_max_seq,])
        dec_embed = Embedding(self.num_classes,self.embed_dim)(dec_input)
        decoder = LSTM(50,return_sequences=True)
        dec_output = decoder(dec_embed,initial_state=[hs,cs])
        
        dec_output = Dense(self.num_classes,activation='softmax')(dec_output)
        self.model = Model(inputs=[enc_input,dec_input], outputs=dec_output)        
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    
    def train(self, data_generator, num_epochs):
        self.model.fit_generator(generator = data_generator,
                                   steps_per_epoch = 200,
                                   epochs = num_epochs)
    
    def predict(self, input_sample):
        return self.model.predict(input_sample)
    
    
class train_gen:
    def __init__(self,num_samples):
        self.ns = num_samples
        self.read_files()
        self.int_mapping()
        self.ohe = OneHotEncoder(n_values=self.num_vocab,sparse=False)
        
    def read_files(self):
        f = open('letters_source.txt')
        self.source_list = f.read().splitlines()
        f.close()
        f = open('letters_target.txt')
        self.target_list = f.read().splitlines()
        f.close()
        self.len_max_seq = max([len(i) for i in self.source_list])
        self.len_list = len(self.source_list)
    
    def int_mapping(self):
        self.unique_char = list(set(list(''.join(self.source_list))))
        self.unique_char += ['<st>','<end>','<pad>']
        self.num_vocab = len(self.unique_char)
        self.char_to_int = {}
        self.int_to_char = {}
        for i in range(len(self.unique_char)):
            self.char_to_int[self.unique_char[i]] = i
            self.int_to_char[i] = self.unique_char[i]
    
    def get_one_sample(self):
        idx = random.randint(0,self.len_list-1)
        data_enc = list(self.source_list[idx])
        data_enc = ['<pad>']*(self.len_max_seq-len(data_enc)) + data_enc 
        data_enc = [self.char_to_int[i] for i in data_enc]
        data_dec = ['<st>'] + ['<pad>']*(len(data_enc)-1)
        data_dec = [self.char_to_int[i] for i in data_dec]
        
        target = list(self.target_list[idx])
        target = target + ['<pad>']*(self.len_max_seq-len(target))
        target = [self.char_to_int[i] for i in target]
        
        return data_enc, data_dec, target
    
    def gen(self):
        while(1):
            data_enc = np.zeros([self.ns,self.len_max_seq,])
            data_dec = np.zeros([self.ns,self.len_max_seq,])
            target = np.zeros([self.ns,self.len_max_seq,self.num_vocab])
            for i in range(self.ns):
                de,dd,t = self.get_one_sample()
                data_enc[i] = de
                data_dec[i] = dd
                t = np.expand_dims(np.array(t),axis=1)
                t = self.ohe.fit_transform(t)
                target[i] = t
            yield [data_enc, data_dec], target
    
    def predict_on_input(self,net,text):
        data_enc = list(text)
        data_enc = ['<pad>']*(self.len_max_seq-len(data_enc)) + data_enc 
        data_enc = [self.char_to_int[i] for i in data_enc]
        data_enc = np.array(data_enc)
        data_enc = np.expand_dims(data_enc,axis=0)
        data_dec = ['<st>'] + ['<pad>']*(data_enc.shape[1]-1)
        data_dec = [self.char_to_int[i] for i in data_dec]
        data_dec = np.array(data_dec)
        data_dec = np.expand_dims(data_dec,axis=0)
        sorted_text = list(np.argmax(net.predict([data_enc,data_dec]),axis=2)[0])
        sorted_text = [self.int_to_char[i] for i in sorted_text]
        return sorted_text

if __name__ == "__main__":        
    batch_size = 64
    embed_dim = 16
    num_epochs = 20
    tg = train_gen(batch_size)
    tg_gen = tg.gen()
    net = LSTMSeq2Seq(tg.num_vocab,tg.len_max_seq,16)
    net.train(tg_gen,num_epochs)
    st = tg.predict_on_input(net,'gftghy')
    print(st)
