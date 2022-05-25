import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 使用functional API
def lenet5(num_classes):
     return tf.keras.Sequential([
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(224, 224, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(num_classes, activation="softmax")       
    ])
    