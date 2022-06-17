import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ReLU

def lenet5(input, num_classes):

    x = Conv2D(32, (5, 5), strides=2, padding="same")(input)
    x = ReLU()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(16, (3, 3), padding="same")(x)
    x = ReLU()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Flatten()(x)

    x = Dense(num_classes, activation="softmax")(x)

    return x