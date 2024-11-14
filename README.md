# Image Classification Model with TensorFlow

This project is a Convolutional Neural Network (CNN) model built using TensorFlow and Keras. It is designed to classify images into one of three categories, utilizing data preprocessing, augmentation, and a series of convolutional and dense layers for feature extraction and classification.

## Project Structure

- `model.py`: The main script that defines and builds the CNN model.
- `requirements.txt`: Lists the required Python packages to run the project.
- `README.md`: Provides an overview of the project.

## Model Architecture

The model architecture consists of several convolutional, pooling, and dense layers. Here’s an overview of the layers used:

1. **Preprocessing Layers**
   - `resize_and_rescale`: Resizes and rescales the input images.
   - `data_augmentation`: Random augmentations are applied to input images to improve generalization.

2. **Convolutional and Pooling Layers**
   - 6 Convolutional layers with ReLU activation.
   - Max Pooling layers to reduce spatial dimensions and parameters.

3. **Flatten and Dense Layers**
   - A Flatten layer to convert 2D features to a 1D vector.
   - A Dense layer with 64 units and ReLU activation.
   - A final Dense layer with a softmax activation function for multi-class classification.

## Requirements

Install the required packages by running:

```bash
pip install -r requirements.txt
```

## Model Summary

The model is built with the following specifications:
- **Input Shape**: `(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)`
- **Number of Classes**: `3`

## Example Usage

```python
import tensorflow as tf
from model import model  # Assuming your model definition is saved as model.py

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy}")
```

## Visualization

Use `matplotlib` to plot training and validation metrics over epochs.

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## Notes

- This model is flexible and can be adjusted for more classes by modifying the `n_classes` parameter.
- For image preprocessing and augmentation, TensorFlow layers like `resize_and_rescale` and `data_augmentation` are utilized.
 overview and guide for using your model. Let me know if you’d like additional sections!
