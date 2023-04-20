For me to better understand the Transformer architecture and how to train them, I designed this simple sequence2sequence example which takes in a string and sorts in alphabetical order. 
I had done something similar when I was initially learning about RNNs [here](https://github.com/prith189/DeepLearning/blob/master/Seq2Seq/Train.py)
  
It learns to do the job well within 4 epochs. Here is the output on unseen data after 4 epochs:

```
Input string: sortingtest - Sorted string: eginorssttt
Input string: abcdef - Sorted string: abcdef
Input string: fedcba - Sorted string: abcdef
Input string: prithvi - Sorted string: hiiprtv
```
