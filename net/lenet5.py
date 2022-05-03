import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def lenet5(num_classes, input_shape):
    return tf.keras.Sequential([
        Conv2D(20, (5, 5), padding="same", input_shape=input_shape, activation="relu"),
        MaxPooling2D(2, 2), 
        Conv2D(50, (5, 5), padding="same", activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(500, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])