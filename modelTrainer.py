import idx2numpy
from tensorflow import keras
from keras import layers


# 1. Data Preparation
x = idx2numpy.convert_from_file('./MNIST_ORG/train-images.idx3-ubyte')  # Input data
y = idx2numpy.convert_from_file('./MNIST_ORG/train-labels.idx1-ubyte')  # Output labels

# 2. Model Definition
model = keras.Sequential([
    layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='sigmoid'),  # Hidden layer (Only one hidden layer because convolutional layers extract all the features)
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax'),  # Output layer
])
modelNumber = 28

# 3. Model Compilation
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Model Training
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='./models/mnist_number_predictor_epochs/mnist_number_predictor_model_{epoch:02d}.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    save_freq='epoch'
)
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    min_delta=0.005,
    restore_best_weights=True)

epochCount = 50
model.fit(x, y, epochs=epochCount, verbose=1, callbacks=[checkpoint_callback, early_stop])

# 5. Save Model
model.save(f'./models/mnist_number_predictor_model{modelNumber}_{epochCount}epochs.keras')
