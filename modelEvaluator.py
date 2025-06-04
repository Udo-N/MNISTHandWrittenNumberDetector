import idx2numpy
import numpy as np
from tensorflow import keras

# 1. Load model
fpath = './models/mnist_number_predictor_model28_50epochs.keras'
model = keras.models.load_model(fpath)

# 2. Load Data
testData = idx2numpy.convert_from_file('./MNIST_ORG/t10k-images.idx3-ubyte')

# 3. Determine accuracy
matches = np.argmax(predictions, axis=1) == testLabels
accuracy = np.mean(matches) * 100
print(f"Testing Accuracy: {accuracy:.2f}%")

loss, trainingAccuracy = model.evaluate(idx2numpy.convert_from_file('./MNIST_ORG/train-images.idx3-ubyte'), idx2numpy.convert_from_file('./MNIST_ORG/train-labels.idx1-ubyte'))
print(f"Training Accuracy: {(trainingAccuracy*100):.2f}%")
