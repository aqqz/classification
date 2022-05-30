import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ReLU

def lenet5(input, num_classes):

    x = Conv2D(32, (3, 3), padding="same")(input)
    # x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    # x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    # x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(2, 2)(x)

    x = Flatten()(x)

    x = Dense(num_classes, activation="softmax")(x)

    return x