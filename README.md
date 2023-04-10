# DeepLearning
Sharing some personal ML projects I've worked on
1. [Neural Network from first principles (using only Numpy)](https://github.com/prith189/DeepLearningShowcase/tree/master/FirstPrinciples)
      - Implemented a Fully connected dense layer architecture using just simple Numpy with functions to backpropagate and update gradients 
      - Trained on MNIST dataset using Fully connected layers with reasonable accuracy (~98%)
2. Recurrent Neural Networks
      - [Sequence2Sequence Network](https://github.com/prith189/DeepLearning/tree/master/Seq2Seq) using Keras using the Encoder <-> Decoder architecture and trained on a simple task of sorting characters. For eg. 'adgbf' would output the sorted sequence 'abdfg'. This architecture can be easily expanded to do language translation for eg.
      - [CharacterLevelRNN](https://github.com/prith189/DeepLearningShowcase/tree/master/RecurrentNetwork) Trained a simple character level RNN on a random C code base
3. Convolutional Neural Networks
      - [Carvana Image Segmentation](https://github.com/prith189/DeepLearning/tree/master/Image_Segmentation) Kaggle contest to segment the outline of a car in an image. Used the UNet architecture and built in Keras
      - [Yelp Restaurant Photo Classification](https://github.com/prith189/Yelp_Restaurant_Photo_Classification) 16th placed solution to classify crowd sourced images of restaurants
      - [Kuzushiji Character Recognition](https://github.com/prith189/DeepLearning/tree/master/Kuzushiji) 22nd place in a contest to read characters from 1000 year old Kuzushiji documents. Source code to be uploaded soon
4. Generative Adverserial Networks
      - [Speech Sample Generation](https://github.com/prith189/DeepLearning/tree/master/Speech_GAN) Using Small Speech samples, built a GAN that can generate speech. Built using Keras on a dataset provided by Google
