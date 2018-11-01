# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 20:01:32 2018

@author: prithvi
"""
import numpy as np
from keras.layers import Dense, Input, BatchNormalization, LSTM, Flatten, LeakyReLU, Reshape, Conv2D, Conv2DTranspose, Activation
from keras.optimizers import Adam
from keras.models import Model
from Spectrogram_gen import Datagen


class DCGAN:
    def __init__(self):
        
        self.latent_dim = 100
        self.im_width = 48
        self.im_height = 64
        self.im_chan = 1
        self.batch_size = 32
        
        self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
        
        self.build_generator()
        
        noise = Input([self.latent_dim,])
        gen_img = self.generator(noise)
        self.discriminator.trainable = False
        pred = self.discriminator(gen_img)
        self.stacked = Model(noise,pred)
        self.stacked.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
        
        self.data = Datagen('happy')
        self.dgen = self.data.batch_gen(self.batch_size)
        
        pass
    
    def build_generator(self):
        inp = Input([self.latent_dim,])
        x1 = Dense(128*8*6)(inp)
        x1 = Activation('relu')(x1)
        x1 = Reshape([8,6,128])(x1)
        x1 = Conv2DTranspose(64,kernel_size=3,strides=2,padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2DTranspose(32,kernel_size=3,strides=2,padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2DTranspose(16,kernel_size=3,strides=2,padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2DTranspose(self.im_chan,kernel_size=3,strides=1,padding='same')(x1)
        y = Activation('tanh')(x1)
        self.generator = Model(inp,y)
        
    def build_discriminator(self):
        inp = Input([self.im_height,self.im_width,self.im_chan])
        x1 = Conv2D(32,kernel_size=3,strides=2,padding='same')(inp)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = Conv2D(64,kernel_size=3,strides=2,padding='same')(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = Conv2D(128,kernel_size=3,strides=2,padding='same')(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = Conv2D(256,kernel_size=3,strides=2,padding='same')(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = Flatten()(x1)
        x1 = Dense(64)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        y = Dense(1,activation='sigmoid')(x1)
        self.discriminator = Model(inp,y)
    
    def train_discriminator(self):
        gen_imgs = self.gen_images(int(self.batch_size))
        true_imgs = next(self.dgen)
        np_ones = np.ones([self.batch_size,])
        np_zeros = np.zeros([self.batch_size,])
        loss = self.discriminator.train_on_batch(gen_imgs,np_zeros)
        loss += self.discriminator.train_on_batch(true_imgs,np_ones)
        return loss
        
    def gen_images(self,num):
        noise_arr = np.random.normal(0,1,[num,self.latent_dim])
        gen_imgs = self.generator.predict(noise_arr)
        return gen_imgs
    
    def train_stacked(self):
        feature_arr = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
        target_arr = np.ones([self.batch_size,])
        loss = self.stacked.train_on_batch(feature_arr,target_arr)
        return loss
    
    def train_GAN(self,epochs):
        for ep in range(epochs):
            d_loss = self.train_discriminator()
            g_loss = self.train_stacked()
            print(ep, d_loss, g_loss)
    
        
if __name__ == '__main__':
    gan = DCGAN()
    gan.train_GAN(4000)
    cf = gan.gen_images(64)
    for j in range(64):
        gan.data.spectrogram_to_audio(cf[j])
