#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 22:24:04 2018

@author: prithvi
"""
from SP_Functions import *
import numpy as np
from scipy.io import wavfile
import random
import os

fft_size = 512 # window size for the FFT
step_size = 32 # distance to slide along the window (in time)
spec_thresh = 4 # threshold for spectrograms (lower filters out more noise)
lowcut = 500 # Hz # Low cut for our butter bandpass filter
highcut = 15000 # Hz # High cut for our butter bandpass filter

n_mel_freq_components = 64 # number of mel frequency channels
shorten_factor = 10 # how much should we compress the x-axis (time)
start_freq = 0 # Hz # What frequency to start sampling our melS from 
end_freq = 8000 # Hz # What frequency to stop sampling our melS from 
num_iter = 10

mel_filter, mel_inversion_filter = create_mel_filter(fft_size = fft_size,n_freq_components = n_mel_freq_components,samplerate = 16000,start_freq = start_freq,end_freq = end_freq)

def one_second_frame(frame):
    #If the input wav file is not eq to 16000 samples, append zeros or cut the samples
    output_frame = np.zeros([16000,],dtype=np.int32)
    if(frame.shape[0]>=16000):
        delta = int((frame.shape[0]-16000)/2)
        output_frame = frame[delta:delta+16000]
        
    if(frame.shape[0]<16000):
        delta = int((16000 - frame.shape[0])/2)
        output_frame[delta:delta+frame.shape[0]] = frame
    
    output_frame = output_frame.astype(np.int16)
    return output_frame

    
def to_spectrogram(frame):
    input_wav = one_second_frame(frame)
    wav_spectrogram = pretty_spectrogram(input_wav.astype('float64'), fft_size = fft_size, 
                                   step_size = step_size, log = True, thresh = spec_thresh)
    mel_spec = make_mel(wav_spectrogram, mel_filter, shorten_factor = shorten_factor)                 
    return mel_spec


def scale(x):    
    x = x/4.1    
    return x

def unscale(x):
    x = x*4.1
    return x
    
class Datagen:
    def __init__(self,word_to_gen):
        self.word = word_to_gen
        self.file_path = './train/audio/'+self.word
        self.wav_files = os.listdir(self.file_path)
        self.gen_num = 0
        
    def batch_gen(self,batch_size):
        while(1):
            feat_array = np.zeros([batch_size,64,48,1])
            for i in range(batch_size):
                wf = random.choice(self.wav_files)
                file_to_read = self.file_path+'/'+wf
                _,frame= wavfile.read(file_to_read)
                feat_array[i,:,:,0] = scale(to_spectrogram(frame))
            yield feat_array
    
    def spectrogram_to_audio(self,gen_sample):
        mel_spec = unscale(gen_sample[:,:,0])
        mel_inverted_spectrogram = mel_to_spectrogram(mel_spec, mel_inversion_filter,spec_thresh=spec_thresh,shorten_factor=shorten_factor)
        recovered_audio_orig = invert_pretty_spectrogram(np.transpose(mel_inverted_spectrogram), fft_size = fft_size,step_size = step_size, log = True, n_iter = num_iter)
        recovered_audio_orig = recovered_audio_orig*32767*20.
        recovered_audio_orig = recovered_audio_orig.astype(np.int16)
        file_to_write = self.file_path + '/gen_sample_' + str(self.gen_num) + '.wav'
        wavfile.write(file_to_write,16000,recovered_audio_orig)
        self.gen_num += 1
