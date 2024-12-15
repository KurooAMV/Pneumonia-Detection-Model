Pneumonia is a serious lung infection that causes the air sacs in the lungs to fill with fluid or pus, making it difficult to breathe.
As many times, pneumonia infection is treated as normal cold and cough, it goes undetected until it lasts longer.

PROPOSED WORK:
This project utilizes a Python-based AI model to detect pneumonia from chest X-rays.
It leverageds CNN to build, train and evaluate a robust classification model.

KEY FEATURES:
Inputs: Chest X-ray images
Outputs: Predictions indicating whether the X-ray shows signjns odf pneumonia or is normal
Libraries Used: 
Tensorflow, Keras - Model creation
Sklearn - Data processing
Seaborn, Matplotlib - Visualization
Numpy - numerical operations

DATASET: 
The data set is leveraged from Kaggle consisting over 5,800 labelled chest x-rays images:
Divided in to train, test and validation set and further into Normal and Pneumonia classes

EVALUATION METRICS:
AUC Score = 97.98%
Recall Score = 98.72%
