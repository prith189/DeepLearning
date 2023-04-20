---
layout: default
---

# About

Hello! My name is Prithvi Nuthanakalva, and I'm currently a Machine Learning Engineer at a startup. In my previous career, I helped build and ship GPS Positioning and Navigation software, and have extensive experience in a technical customer facing role. 

Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/pnuthanakalva/)

# Website

As a personal project, I recently built an end to end ML application which uses a ML model to provide fantasy recommendations for cricket games based on the players involved, also provides odds of a team winning. For every game, predictions are available 30 minutes prior to game start time (100s of simualtions are run in the background once the teams are announced). The website is live at [betmancric.com](https://betmancric.com)


# Projects

Sharing some personal ML projects I've worked on <br /> <br />
1. Language Models <br />
      - [Simple Seq2Seq using Transformers (2023)](https://github.com/prith189/DeepLearning/blob/master/TransformerSeq2Seq/TransformerSorting.ipynb) <br />
        - To get a better understanding of how Transformers work, I experimented with a simple example to sort strings using Transformers. Trained a network using Pytorch
      - [Patient Clinical Notes (2022)](https://github.com/prith189/DeepLearning/blob/master/Clinical_PatientNotes_NBME/NBME.ipynb) <br />
        - Finetuned a Huggingface BERT model in two stages (a) Masked Language Modeling task using the unlabeled patient notes and (b) Named Entity Recognition task using the labeled Description -> Symptom dataset

2. [Neural Network from first principles using only Numpy (2018)](https://github.com/prith189/DeepLearningShowcase/tree/master/FirstPrinciples) <br />
      - Implemented a Fully connected dense layer architecture using just simple Numpy with functions to backpropagate and update gradients <br />
      - Trained on MNIST dataset using Fully connected layers with reasonable accuracy (~98%) <br /> <br />
3. Recurrent Neural Networks <br />
      - [Simple Sequence2Sequence Network using RNNs (2018)](https://github.com/prith189/DeepLearning/tree/master/Seq2Seq) 
         - Built using Keras using the Encoder <-> Decoder architecture and trained on a simple task of sorting characters. For eg. 'adgbf' would output the sorted sequence 'abdfg'. This architecture can be easily expanded to do language translation for eg. <br />
      - [CharacterLevelRNN (2018)](https://github.com/prith189/DeepLearningShowcase/tree/master/RecurrentNetwork) 
        - Trained a simple character level RNN on a random C code base <br /> <br />
4. Convolutional Neural Networks <br />
      - [Carvana Image Segmentation (2018)](https://github.com/prith189/DeepLearning/tree/master/Image_Segmentation) 
        - Kaggle contest to segment the outline of a car in an image. Used the UNet architecture and built in Keras <br />
      - [Yelp Restaurant Photo Classification (2018)](https://github.com/prith189/Yelp_Restaurant_Photo_Classification)
        - 16th placed solution to classify crowd sourced images of restaurants <br />
      - [Kuzushiji Character Recognition (2020)](https://github.com/prith189/DeepLearning/tree/master/Kuzushiji) 
        - 22nd place in a contest to read characters from 1000 year old Kuzushiji documents. Source code to be uploaded soon <br /> <br />
5. Generative Adverserial Networks <br />
      - [Speech Sample Generation (2018)](https://github.com/prith189/DeepLearning/tree/master/Speech_GAN) 
        - Using Small Speech samples, built a GAN that can generate speech. Built using Keras on a dataset provided by Google <br />

