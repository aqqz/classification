from uuid import RESERVED_FUTURE
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, \
    BatchNormalization, ReLU, Concatenate, Dropout, AveragePooling2D

def ResidualBlock(input, filters, strides=1):
    x = Conv2D(filters, (3, 3), padding="same", strides=strides)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)
    x = BatchNormalization()(x)

    input = Conv2D(filters, (1, 1), strides=strides)(input)
    out = input + x
    return ReLU()(out)


def resnet(input, num_classes):
    # input_shape (224, 224, 3)
    x = Conv2D(64, (7, 7), strides=2, padding="same")(input)
    
    x = MaxPooling2D((3, 3), strides=2, padding="same")(x)
    
    x = ResidualBlock(x, 64)
    x = ResidualBlock(x, 64)
    x = ResidualBlock(x, 64)
    x = ResidualBlock(x, 64)

    x = ResidualBlock(x, 128, strides=2)
    x = ResidualBlock(x, 128)
    x = ResidualBlock(x, 128)
    x = ResidualBlock(x, 128)

    x = ResidualBlock(x, 256, strides=2)
    x = ResidualBlock(x, 256)
    x = ResidualBlock(x, 256)
    x = ResidualBlock(x, 256)

    x = ResidualBlock(x, 512, strides=2)
    x = ResidualBlock(x, 512)
    x = ResidualBlock(x, 512)
    x = ResidualBlock(x, 512)
    
    x = AveragePooling2D((7, 7), strides=1)(x)
    x = Flatten()(x)
    return Dense(num_classes, activation="softmax")(x)