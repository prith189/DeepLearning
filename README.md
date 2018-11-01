# DeepLearning
Source code for some deep learning related projects that I've implemented
1. [Neural Network from first principles (using only Numpy)](https://github.com/prith189/DeepLearningShowcase/tree/master/FirstPrinciples)
      - Implemented a Fully connected dense layer architecture with functions to backpropagate and update gradients using only Numpy
      - Trained a MNIST dataset using Fully connected layers with reasonable accuracy (~98%)
2. Recurrent Neural Networks
      - [Sequence2Sequence Network](https://github.com/prith189/DeepLearningShowcase/tree/master/FirstPrinciples) using Keras using the Encoder <-> Decoder architecture and trained on a simple task of sorting characters. For eg. 'adgbf' would output the sorted sequence 'abdfg'. This architecture can be easily expanded to to language translation for eg.
      - [CharacterLevelRNN](https://github.com/prith189/DeepLearningShowcase/tree/master/RecurrentNetwork) Trained a simple character level RNN on a random C code base
3. Convolutional Neural Networks
      - [Carvana Image Segmentation]() Kaggle contest to segment the outline of a car in an image. Used the UNet architecture and built in Keras
4. Generative Adverserial Networks
      - [Speech Sample Generation](https://github.com/prith189/DeepLearning/tree/master/Speech_GAN) Using Small Speech samples, built a GAN that can generate speech. Built using Keras on a dataset provided by Google
