# Big Data ML Pipeline 2162940

## Introduction
This project will provide a comprehensive report overviewing the steps required to deploy a Machine Learning Pipeline for Image Classification, employing Convolutional Neural Network (CNN). The entire process will be documented, from the data collection stage, through to the building of the model, evaluating its performance and using this to predict future performances. This will be repeated for three different models, which continue to improve upon the last in an iterative process. The structure of the code, written in a separate Jupyter Notebook, will be covered before potential future improvements will be considered - these will be influenced by the unfixed bugs that are present in the model.

The data set used in this model is the Modified National Institute of Standards and Technology, abbreviated to MNIST, which is a classic dataset used for handwritten digit classification. It contains greyscale images of digits 0-9, each 28x28 pixels in size.

## Business Objectives
The aim of this project is to develop a ML Pipeline model which is capable of predicting handwritten digit classes with accuracies greater than or equal to 90% on unseen data.

This will be achieved by first developing a model and then using iterative methods to continually improve its accuracy and efficiency, whilst simultaneously fixing any bugs that are encountered during the process. By the end of this process, the model should meet its target and be easy to understand, allowing for others to improve upon it in the future.

## ML Pipeline
The following section is split into several subsections which tell the story of the Machine Learning Pipeline, before documenting how the dataset will be explored in an Exploratory Data Analysis subsection. Then, the building and evaluation of three different models will be covered, including the prediction of their success rates.

### 1. Data Collection
The MNIST dataset was imported using the TensorFlow Keras API:

```python
from tensorFlow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, X_test.shape)
```

The above code loads the dataset and splits it into training (60,000 images) and testing (10,000 images) subsets.

The data was then validated by inspecting shapes, types and values, ensuring there were no missing entries. Pixel values were normalised to the range [0, 1] and reshaped into 3D tensors for CNN input.

To split the data into training, validation and test subsets, the original 60,00 training images that MNIST had previously split was divided again into the new training (54,000 images) and validation (6,000 images) and the test set was left only to be used after the model was trained for a final performance evaluation.

### 2. Exploratory Data Analysis (EDA)
An EDA was conducted in order to better understand the structure and content of the MNIST dataset. Key points include analysing the image data and investigating the label distribution.

In order to verify that the images are loaded correctly, some random images can be plotted:

```python
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {np.argmax(y_train_cat[i])}")
    plt.axis('off')
plt.suptitle("Sample MNIST Digits")
plt.show()
```

The above code generates 10 random integers and then plots the corresponding images in the training subset from the MNIST dataset. They are displayed in a greyscale 28x28 pixel format.

In order to check for imbalances, the distribution of digit labels is plotted:

```python
unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique, counts)
plt.title("Distribution of Digits in Training Set")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.show()
```

This code plots a bar chart showing the frequency of each digit in order to verify no imbalances were present in the dataset.

### 3. Model Development
The following section details the iterative process of building the CNN model, showing where the improvements were made along the way.

#### First Model
The first model that was built was a basic CNN, designed to be simple whilst still performing at a viable level.

##### Model Building
A simple CNN with one convolutional and pooling layer followed by a dense classifier was used as the baseline model. 

```python
model_1 = Sequential([
  Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
  MaxPooling2D(pool_size)=(2,2)),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(10, activation='softmax')
])
```
The selected epoch number for this model was 5 in order to give a quick feedback loop and to not overfit on the small, one-layer model. This proved to be a benchmark for further iterations of the CNN.

```python
model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy'])
history_1 = model_1.fit(X_train, y_train_cat, epochs=5, batch_size=128, validation_data=(X_val, y_val_cat))
```

##### Model Evaluation
The model had a validation accuracy of approximately 98.3%, which is viable based on the previously stated objectives. It was fast to train and simple to understand. It is likely that after the first 5 epochs, the validation loss would plateau or possibly diverge in such a simple model.

The following code was used to provide a visual representation of the training curve of each model:

```python
plt.plot(history_1.history['accuracy'], label='Train Accuracy')
plt.plot(history_1.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy over Epochs")
plt.legend()
plt.show()
```

#### Second Model
The second iteration was a deeper CNN with Dropout which made improvements based on the first model's limitations.

##### Model Building
The second model added another convolutional layer and dropout regularisation, which was implemented to help reduce overfitting.

```python
model_2 = Sequential([
  Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
  MaxPooling2D((2,2)),
  Conv2D(64, (3,3), activation='relu'),
  MaxPooling2D((2,2)),
  Flatten(),
  Dense(128, activation='relu'),
  Dropout(0.5),
  Dense(10, activation='softmax')
])
```

The chosen epoch number for this model was 10.

##### Model Evaluation
The second CNN had an increased validation accuracy of roughly 99.2% and it handled the variation better. There was also less issues in regard to overfitting however the training time was slightly longer.

There was still room for improvement in the area of accuracy and generalisation.

#### Third Model
The third model was developed as an optimised CNN with Batch Normalisation to further increase its depth.

##### Model Building
The third model was designed to maximise classification accuracy while maintaining training stability and generalisation. Batch normalisation and an increased dense layer size were implemented.

```python
model_3 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```
This model had an epoch number of 12.

##### Model Evaluation
The accuracy of this model further improved up to 99.3% and there was a minimal gap between the training and validation accuracy. This demonstrates a great level of generalisation and no major overfitting was observed.

The downsides are that it took longer to train and had a higher memory usage.

### 4. Prediction
Once the third model was trained and validated, it was used to make predictions on unseen data. This was carried out to test the model's ability to generalise beyond the training and validation subsets.

The following code was used:
```python
# Generate predictions for the test set
y_pred = model_3.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
```
This model outputted a ten class probability vector for each test image and predicted the class with the highest probability. The final preciction was compared against the true label to evaluate its performance.

To visualise these predictions, random samples from the test set were selected and annotated with the predicted and actual digit values:
```python
# Visualise a few predictions
plt.figure(figsize=(12, 4))
for i in range(10):
    idx = np.random.randint(0, X_test.shape[0])
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {y_pred_classes[idx]} | True: {y_test[idx]}")
    plt.axis('off')
plt.suptitle("Sample Predictions from Test Set")
plt.show()
```
The overall performance of this model was then evaluated using classification metrics:
```python
print("Classification Report:\n")
print(classification_report(y_test, y_pred_classes))

print("Confusion Matrix:\n")
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred_classes), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix on Test Set')
plt.show()
```


## Jupyter Notebook Structure
The Jupyter Notebook was arranged into logical sections as follows:

### 1. Library Imports
### 2. Data Loading and Preprocessing
### 3. EDA
### 4. First Model
### 5. Second Model
### 6. Third Model
### 7. Predictions
### 8. Performance Evaluation

## Future Work
While the final model performed well, there are areas which could be further developed in the future. The implementation of early stopping and learning rate schedulers could further improve on the overfitting issue and allow the model to generalise better to unseen dataa.

There is also room for further development of the model into a real-world environment, such as a mobile app or a web interface, which would allow users to classify digits in real-time.

A more advanced neural network architecture, such as ResNet or MobileNet, could lead to further improvements in accuracy and training efficiency.

These developments would simultaneously improve the technical performance of the system and expand its utilisation in practical settings.

## Libraries and Modules
The following libraries were used in the Jupyter Notebook:

NumPy – Efficient numerical operations and array manipulations;
Matplotlib/Seaborn – Data visualization: plotting digit samples, confusion matrices, and training curves;
TensorFlow/Keras – High-level neural network API used for building and training CNNs;
sklearn – Splitting datasets, confusion matrix, classification metrics;
Pandas – (Minimal use, for dataframe-like manipulations if needed);
Plotly.express - For figure generation.

## Unfixed bugs
Occasionally, the val_accuracy slightly fluctuated due randomness in data split or training.

## Acknowledgements and References
Code snippets adapted with assistance from OpenAi ChatGPT

Dataset provided by TensorFlow Datasets - MNIST

## Conclusions
This project has demonstarted how to build a Machine Learning pipeline using a Convolutional Neural Network for handwritten digit classification on the MNIST dataset. Using an iterative process, three models were constructed, with each improving on the last in terms of accuracy and generalisation. The final model had an accuracy of 99.3%, surpassing the stated goal of >90% in the business objectives.

The models were trained using standard preprocessing steps and the pipeline includes data normalisation, one-hot encoding of labels, validation splitting and multiple performance evaluations, which have also been visually represented.

Overall, this pipeline demonstrates the effectiveness of CNNs in image classification and provides a sound foundation for future developments on these models or in the wider field.
