import os
import streamlit as st
import random
import numpy as np 
import matplotlib.pyplot as plt
# import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
# from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras
from keras_preprocessing.image import ImageDataGenerator
import time
from PIL import Image

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
   
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / np.sum(cm)
    precision = TP / (TP+FP)
    recall = TN/ (TP + FN)
    specificity = TN / (TN +FP)
    f1= 2*precision*recall/(precision+recall)
    st.write(f"[Summary Statistics] \nAccuracy = {accuracy:.2%}\nPrecision = {precision:.2%}\nRecall = {recall:.2%}\nSpecificity = {specificity:.2%}\nF1 Score = {f1:.2%}")
    st.pyplot(fig)    
    
    plt.tight_layout()
    
    
trainPath = "C:/Laptop remains/STUTI/Programa/Stock Predictor/Python programs/Projects/PneumoniaDetectionModel/chest_xray/train/"
valPath = "C:/Laptop remains/STUTI/Programa/Stock Predictor/Python programs/Projects/PneumoniaDetectionModel/chest_xray/val/"
testPath = "C:/Laptop remains/STUTI/Programa/Stock Predictor/Python programs/Projects/PneumoniaDetectionModel/chest_xray/test/"

st.title("Pneumonia Detector using CNN")
st.sidebar.header("Model Settings")
# trainPath = st.sidebar.text_input("Training Data path",trainPath)
# testPath = st.sidebar.text_input("Test Data path",testPath)
# valPath = st.sidebar.text_input("Validation Data path",valPath)
dimen = st.sidebar.slider("Image Dimensions",32,128,64)
batchS = int(st.sidebar.number_input("Batch Size", 32, 512, 128))
epochs = int(st.sidebar.number_input("Epochs",1,200,10))
# channels = st.sidebar.selectbox("Color Mode", [1, 3], index=0)
# mode = 'grayscale' if channels == 1 else 'rgb'
st.sidebar.text_input("Color","Grayscale (1 channel)")

startTime = time.time()
seedValue = 42
os.environ['PYTHONHASHSEED'] = str(seedValue)
random.seed(seedValue)
np.random.seed(seedValue)
tf.random.set_seed(seedValue)

# trainPath = "C:/Laptop remains/STUTI/Programa/Stock Predictor/Python programs/PneumoniaDetectionModel/chest_xray/train/"
# valPath = "C:/Laptop remains/STUTI/Programa/Stock Predictor/Python programs/PneumoniaDetectionModel/chest_xray/val/"
# testPath = "C:/Laptop remains/STUTI/Programa/Stock Predictor/Python programs/PneumoniaDetectionModel/chest_xray/test/"

# dimen = 64
# batchS = 128
# epochs = 100
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

if not os.path.exists(trainPath):
    st.error(f"Training data directory does not exist: {trainPath}")
else:
    trainGenerator = trainDataGen.flow_from_directory(directory=trainPath,
                                                    target_size = (dimen,dimen),
                                                    batch_size = batchS,
                                                    color_mode = mode,
                                                    class_mode = 'binary',
                                                    seed = 42
                                                    )
if not os.path.exists(valPath):
    st.error(f"Validation data does not exists: {valPath}")
else:
    valGenerator = valDataGen.flow_from_directory(directory = valPath,
                                                target_size = (dimen,dimen),
                                                batch_size = batchS,
                                                class_mode = 'binary',
                                                color_mode = mode,
                                                shuffle = False,
                                                seed = 42
                                                )

if not os.path.exists(testPath):
    st.error(f"Testing data does not exist: {testPath}")
else:
    testGenerator = testDataGen.flow_from_directory(directory = testPath,
                                                    target_size = (dimen,dimen),
                                                    batch_size = batchS,
                                                    class_mode = 'binary',
                                                    color_mode = mode,
                                                    shuffle  = False,
                                                    seed = 42
                                                    )

testGenerator.reset()
st.write("### Building the CNN Model")
#Bulding the CNN model
cnn = keras.models.Sequential()
#Layer1
cnn.add(keras.layers.InputLayer(input_shape = (dimen,dimen, channels)))
#Layer 2
cnn.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#layer 3
cnn.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#Layer 4
cnn.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#Layer 5
cnn.add(keras.layers.Flatten())
#Layer 6
cnn.add(keras.layers.Dense(activation='relu',units=128))
cnn.add(keras.layers.Dense(activation='sigmoid',units=1))
#final layer
cnn.compile(optimizer = 'adam', 
            loss = 'binary_crossentropy', 
            metrics = [keras.metrics.AUC()])
if st.button("Train Model"):
    st.write("Model training in Progress...")
    startTime = time.time()
    cnnModel = cnn.fit(
        trainGenerator, 
        steps_per_epoch=len(trainGenerator), 
        epochs=int(epochs), 
        validation_data=valGenerator,
        validation_steps=len(valGenerator), 
        verbose="2")
    endTime = time.time()
    st.success(f"Training completed in {timeInFormat(startTime,endTime)}")
    st.write("### Generating Evaluation Charts...")
    createCharts(cnn,cnnModel,testGenerator)
    # cnnModel.save()
uploaded_file = st.sidebar.file_uploader("Upload an Image for Prediction",type =['jpg','png','jpeg'])

if uploaded_file is not None and st.sidebar.button("Predict"):
    image = Image.open(uploaded_file).convert(mode = 'L')
    image = image.resize((dimen,dimen))
    st.image(image, caption = "Uploaded Image", use_column_width = True)
    image_array = np.array(image)/255.0
    image_array = np.expand_dims(image_array, axis =0)
    prediction = cnn.predict(image_array)
    predicted_class = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1-prediction[0][0]
    st.write(f"### Predicted CLass : {predicted_class}")
    st.write(f"### Confidence : {confidence:.2f}")
