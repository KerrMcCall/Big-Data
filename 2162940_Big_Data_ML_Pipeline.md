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

```python
from tensorFlow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, X_test.shape)
```

The above code loads the dataset and splits it into training (60,000 images) and testing (10,000 images) subsets.

The data was then validated by inspecting shapes, types and values, ensuring there were no missing entries. Pixel values were normalised to the range [0, 1] and reshaped into 3D tensors for CNN input.

To split the data into training, validation and test subsets, the original 60,00 trianing images that MNIST had previously split was divided again into the new training (54,000 images) and validation (6,000 images) and the test set was left oly to be used after the model was trained for a final performance evaluation.

### 2. Exploratory Data Analysis (EDA)
An EDA was conducted in order to better understand the structure and content of the MNIST dataset. Key points include analysing the image data and investigating the label distribution.

In order to verify that the images are loaded correctly, some random images can be plotted:

```python
import matplotlib.pyplot as plt
import numpy as np

# Plot 10 random images from the training dataset
plt.figure(figsize=(10,2))
for i in range(1, 10):
idx = np.random.randint(0, X_train.shape[0])
plt.subplot(1, 10, i + 1)
plt.imshow(X_train[idx[.reshape(28, 28), cmap ='gray'
plt.title(np.argmax(y_train_cat[idx]))
plt.axis('off')
plt.suptitle("Randome Sample Images fromm Training Subset")
plt.show()
```

The above code generates 10 random integers and then plots the corresponding images in the training subset from the MNIST dataset.

In order to check for imbalances, the distribution of digit labels is plotted:

```python
import seaborn as sns
# Count the occurences of each digit in the original training labels
sns.countplot(x=y_train)
plt.title("Distribution of Digit Lables in Training Subset")
plt.xlabel("Digit")
plt.ylabel("Frequency")
plt.show()
```

This code plots the frequency of each digit in order to verify no imbalances in the dataset.

### 3. Model Building

### 4. Model Evaluation

### 5. Prediction

## Jupyter Notebook Structure

## Future Work

## Libraries and Modules

## Unfixed bugs

## Acknowledgements and References

## Conclusions
