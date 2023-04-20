Notebook where I tried a few approaches to solve [this](https://www.kaggle.com/c/nbme-score-clinical-patient-notes) Kaggle problem.

Idea was to use a pretrained BERT model to perform Named Entity recognition - where instead of mapping a word to a entity - we map a phrase in a patient's clinical notes to its corresponding symptom.

However, since a lot of the notes are medical terminology, the model is likely to perform better if we used the medical dataset to finetune the model.

- Use Masked Language Modeling approach to finetune the BERT model
- Use Named Entity recognition using the labeled NBME dataset to further train the model
- Evaluate on unseen data.


Using the two stage finetuning, we see better results than simply finetuning with the labeled dataset.


