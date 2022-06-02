import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ReLU



def net(input, num_classes):
    x = Conv2D(32, (5, 5), strides=2, padding="same")(input)
    x = ReLU()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(16, (3, 3), padding="same")(x)
    x = ReLU()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Flatten()(x)

    out1 = Dense(num_classes, activation="softmax")(x)
    out2 = Dense(4, activation="relu")(x)

    return out1, out2
