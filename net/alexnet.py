import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, ReLU


def alexnet(input, num_classes):
    # input_shape: (227, 227, 3)
    x = Conv2D(96, (11, 11), strides=4)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(3, strides=2)(x)

    x = Conv2D(256, (5, 5), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(3, strides=2)(x)

    x = Conv2D(384, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(384, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(3, strides=2)(x)

    x = Flatten()(x)
    x = Dense(4096)(x)
    x = ReLU()(x)
    x = Dropout(rate=0.5)(x)

    x = Dense(4096)(x)
    x = ReLU()(x)
    x = Dropout(rate=0.5)(x)

    x = Dense(num_classes, activation="softmax")
    return x
