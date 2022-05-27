# Kuzushiji character recognition
Code for this Kaggle contest: https://www.kaggle.com/competitions/kuzushiji-recognition


My final submission which secured 22nd place is a slightly enhanced version of this
![image](https://user-images.githubusercontent.com/9631296/170787953-050d9154-eb64-4b58-8fbf-45c31ba1ffe5.png)




Sample bounding box output:
![image](https://user-images.githubusercontent.com/9631296/170788076-d6282919-7906-4b3b-afa5-0411dacdb516.png)


## Approach
- Expriemnted with a different approach as opposed to the popular YOLO based models
- Centernet architecture to identify bounding boxes of the chracters, and run a MNIST style classification on the bounding box images to identify characters
