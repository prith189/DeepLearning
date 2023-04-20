Notebook where I tried a few approaches to solve [this](https://www.kaggle.com/c/nbme-score-clinical-patient-notes) Kaggle problem.

Idea was to use a pretrained BERT model to perform Named Entity recognition - where instead of mapping a word to a entity - we map a phrase in a patient's clinical notes to its corresponding symptom.

However, since a lot of the notes are medical terminology, the model is likely to perform better if we used the medical dataset to finetune the model.

- Use Masked Language Modeling approach to finetune the BERT model
- Use Named Entity recognition using the labeled NBME dataset to further train the model
- Evaluate on unseen data.


Using the two stage finetuning, we see better results than simply finetuning with the labeled dataset (F1 score of 0.8 vs 0.74)
```
Without MLM finetuning:
Epoch 10/25
7/7 [==============================] - 26s 4s/step
Precision:0.7746124394227423, Recall:0.7256697100431424, F1:0.7493427613090382
400/400 [==============================] - 393s 982ms/step - loss: 1.9812 - accuracy: 0.9471 - val_loss: 0.6336 - val_accuracy: 0.8868

With MLM finetuning:
Epoch 10/25
7/7 [==============================] - 25s 3s/step
Precision:0.7816288283158969, Recall:0.8264272097923147, F1:0.8034040062910717
400/400 [==============================] - 391s 978ms/step - loss: 1.3506 - accuracy: 0.9609 - val_loss: 0.4739 - val_accuracy: 0.9049
```



