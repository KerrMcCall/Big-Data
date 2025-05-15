# Big-Data ML Pipeline 2162940

## Introduction
This project will provide a comprehensive report overviewing the steps required to deploy a Machine Learning Pipeline for Image Classification, employing Convolutional Neural Network (CNN). The entire process will be documented, from the data collection stage, through to the building of the model, evaluating its performance and using this to predict future performances. The structure of the code written in a separate Jupyter Notebook will be covered before potential future improvements will be considered - these will be influenced by the unfixed bugs that are present in the model.

The data set used in this model is the Modified National Institute of Standards and Technology, abbreviated to MNIST, which is a classic dataset used for handwritten digit classification. It contains greyscale images of digits 0-9, each 28x28 pixels in size.

## Business Objectives
The aim of this project is to develop a ML Pipeline model which is capable of predicting handwritten digit classes with accuracies greater than or equal to 90% on unseen data.

This will be achieved by first developing a model and then using iterative methods to continually improve its accuracy and efficiency, whilst simultaneously fixing any bugs and erros that are encountered during the process. By the end of this process, the model should meet its target, run efficiently and be easily followed or copied, allowing for others to improve upon it in the future.

## ML Pipeline
The following section is split into several subsections which tell the story of the Machine Learning Pipeline, before documenting how the dataset will be explored in an Exploratory Data Analysis subsection. Then, the building and evaluation of the model will be covered, before the prediction of its success will be highlighted.

### 1. Data Collection
The MNIST dataset was imported using the TensorFlow Keras API:

'''python
from tensorFlow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, X_test.shape)

The above coee loads thr dataset and splits into training (60,000 images) and testing (10,000 images) subsets.

### 2. Exploratory Data Analysis (EDA)

### 3. Model Building

### 4. Model Evaluation

### 5. Prediction

## Jupyter Notebook Structure

## Future Work

## Libraries and Modules

## Unfixed bugs

## Acknowledgements and References

## Conclusions
