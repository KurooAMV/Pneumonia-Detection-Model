import os
import random
import numpy as np 
import matplotlib.pyplot as plt
# import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import tensorflow as tf
import keras
from keras.regularizers import l2
from keras import metrics
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, InputLayer, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import AUC
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import time

def timeInFormat(startTime,endTime):
    hours, rem = divmod(endTime-startTime, 3600)
    minutes, seconds = divmod(rem, 60)
    return ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def createCharts(cnn,cnnModel,testGenerator):
    trainLoss = cnnModel.history['loss']
    valLoss = cnnModel.history['val_loss']
    
    trainAUCName = list(cnnModel.history.keys())[3]
    valAUCName = list(cnnModel.history.keys())[3]
    trainAUC = cnnModel.history[trainAUCName]
    valAUC = cnnModel.history[valAUCName]
    
    yTrue = testGenerator.classes
    YPred = cnn.predict(testGenerator, steps=len(testGenerator))
    yPred = (YPred>0.5).T[0]
    yPredProb = YPred.T[0]
    
    fig = plt.figure(figsize = (13,20))
    
    #plotting train vs validation loss
    plt.subplot(2,2,1)
    plt.title("Training vs Validation Loss")
    plt.plot(trainLoss, label='training loss')
    plt.plot(valLoss, label='validation loss')
    plt.xlabel("Number of Epochs", size =14)
    plt.legend()
    
    #plotting Train vs validation auc
    plt.subplot(2,2,2)
    plt.title("Training vs Validation AUC Score")
    plt.plot(trainAUC, label = 'train AUC')
    plt.plot(valAUC, label= 'validation AUC')
    plt.xlabel("Number of Epochs", size =14)
    plt.legend()
    
    #plotting the confusion matrix
    plt.subplot(2,2,3)
    cm = confusion_matrix(yTrue,yPred)
    names = ['True Negatives', 'False Negatives' ,'False Positives',
             'True Positives']
    counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    ticklabels = ['Normal', 'Pneumonia']
    
    sns.heatmap(cm,annot = labels, fmt = '', cmap = 'Oranges', 
                xticklabels = ticklabels, yticklabels = ticklabels)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted", size = 14)
    plt.ylabel("Actual", size=14)
    
    #plotting ROC Curve
    plt.subplot(2,2,4)
    fpr, tpr, thresholds = roc_curve(yTrue, yPredProb)
    aucScore = roc_auc_score(yTrue , yPredProb)
    plt.title("ROC Curve")
    plt.plot([0,1],[0,1], 'k--',label="Random(AUC = 50%)")
    plt.plot(fpr,tpr,label='CNN (AUC= {:.2f}%)'.format(aucScore*100))
    plt.xlabel('False Positive Rate', size = 14)
    plt.ylabel('True Positive rate', size =14)
    plt.legend(loc = 'best')
    
    plt.tight_layout()
    
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / np.sum(cm)
    precision = TP / (TP+FP)
    recall = TN/ (TP + FN)
    specificity = TN / (TN +FP)
    f1= 2*precision*recall/(precision+recall)
    print("[Summary Statistics] \nAccuracy = {:.2%}\nPrecision = {:.2%}\nRecall = {:.2%}\nSpecificity = {:.2%}\nF1 Score = {:.2%}".format(accuracy, precision,recall,specificity,f1))
    

    
    
startTime = time.time()
seedValue = 42
os.environ['PYTHONHASHSEED'] = str(seedValue)

random.seed(seedValue)
np.random.seed(seedValue)
tf.random.set_seed(seedValue)

trainPath = "C:/Laptop remains/STUTI/Programa/Stock Predictor/Python programs/PneumoniaDetectionModel/chest_xray/train/"
valPath = "C:/Laptop remains/STUTI/Programa/Stock Predictor/Python programs/PneumoniaDetectionModel/chest_xray/val/"
testPath = "C:/Laptop remains/STUTI/Programa/Stock Predictor/Python programs/PneumoniaDetectionModel/chest_xray/test/"

dimen = 64
batchS = 128
epochs = 100
channels = 1
mode = 'grayscale'

print("Generating Data Set...\n")
trainDataGen = ImageDataGenerator(rescale=1.0/255.0, 
                                  shear_range = 0.2, 
                                  zoom_range = 0.2, 
                                  horizontal_flip = True 
                                  )
valDataGen = ImageDataGenerator(rescale = 1.0/255.0)
testDataGen = ImageDataGenerator(rescale = 1.0/255.0)

trainGenerator = trainDataGen.flow_from_directory(directory=trainPath,
                                                  target_size = (dimen,dimen),
                                                  batch_size = batchS,
                                                  color_mode = mode,
                                                  class_mode = 'binary',
                                                  seed = 42
                                                  )

valGenerator = valDataGen.flow_from_directory(directory = valPath,
                                              target_size = (dimen,dimen),
                                              batch_size = batchS,
                                              class_mode = 'binary',
                                              color_mode = mode,
                                              shuffle = False,
                                              seed = 42
                                              )

testGenerator = testDataGen.flow_from_directory(directory = testPath,
                                                target_size = (dimen,dimen),
                                                batch_size = batchS,
                                                class_mode = 'binary',
                                                color_mode = mode,
                                                shuffle  = False,
                                                seed = 42
                                                )

testGenerator.reset()
print("Building CNN model..\n")
#Bulding the CNN model
cnn = Sequential()
#Layer1
cnn.add(InputLayer(input_shape = (dimen,dimen, channels)))
#Layer 2
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
#layer 3
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
#Layer 4
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
#Layer 5
cnn.add(Flatten())
#Layer 6
cnn.add(Dense(activation='relu',units=128))
cnn.add(Dense(activation='sigmoid',units=1))

#final layer
cnn.compile(optimizer = 'adam', 
            loss = 'binary_crossentropy', 
            metrics = [AUC()])
print("fitting data in the model...")
cnnModel = cnn.fit(
    trainGenerator, 
    steps_per_epoch=len(trainGenerator), 
    epochs=100, 
    validation_data=valGenerator,
    validation_steps=len(valGenerator), 
    verbose="2")
print("Generating charts...")
createCharts(cnn,cnnModel,testGenerator)

endTime = time.time()
print( f"Total time taken: {timeInFormat(startTime,endTime)}" )