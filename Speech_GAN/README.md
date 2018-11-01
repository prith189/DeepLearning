### A Convolutional GAN architecture built in Keras to generate short speech samples ###

While working on this Kaggle competition, since the number of training samples was small, I tried to build synthetic speech samples using a GAN architecture.

The resulting speech samples are fairly decent.

    - Convert speech samples to a spectrogram (basically a 3D array)
    - Use spectrograms for a particular word (for eg. "happy") and train a GAN to generate new spectrogram samples
    - Invert the generated spectrograms to speech samples
 
All the functions for converting speech to spectrograms and vice-versa were borrowed from here: https://github.com/timsainb/python_spectrograms_and_inversion
